"""MA20: joint codec placement, without the legacy positive-score candidate pool.

Every remote contribution is eligible. A bank member is an intervention, not a
claim that its quota is beneficial. Only fresh complete-path GPU calibration
may select a member for an independent full-training comparison.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from common import read, require, validate_entry, write
from ablation_plan import mask_sha

PROTOCOL = 'h100-ma20-codec-first-v1'
C_POLICY = 'interleaved_arrival_selective_attn_mlp'
SHAPES = {'attn': [2048, 4096, 4096], 'mlp': [2048, 4096, 16384]}
DEFAULT_COUNTS = (0, 7, 14, 28, 56, 112)
FIELDS = ('shape', 'M', 'N', 'K_global', 'K_local', 'tiles', 'maps',
          'coords', 'base_coords', 'mask', 'selected_per_source')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def derive(original, count, window=64):
    """Move expensive selected work earlier inside each legal destination window.

    Select from the tail of each source's own original destination order,
    round-robin across destinations. This spreads the intervention across
    receivers and does not deliberately align FP8 fan-in across sources.
    Exchange only selected tail positions and unselected front positions;
    leave the middle of the schedule intact. Logical positions are NOT
    hardware wave IDs or estimates of available overlap.
    """
    require(type(count) is int and count in DEFAULT_COUNTS, 'Unregistered MA20 count')
    require(type(window) is int and window > 0, 'Invalid permutation window')
    entry = {key: copy.deepcopy(original[key]) for key in FIELDS}
    require(entry['tiles'] == 512, 'MA20 requires 512 physical tiles')
    masks, mappings = [], []
    for source in range(8):
        base, old_map, coords = (entry[k][source] for k in ('base_coords', 'maps', 'coords'))
        # All 448 remote contributions are eligible, including legacy zero scores.
        destinations = [(source + offset) % 8 for offset in range(1, 8)]
        queues = {dst: [tile for tile in reversed(coords) if tile // 64 == dst]
                  for dst in destinations}
        candidates = [queues[dst][offset] for offset in range(64) for dst in destinations]
        require(len(candidates) == len(set(candidates)) == 448, 'Incomplete remote candidate universe')
        mask = [0] * 512
        for tile in candidates[:count]:
            mask[tile] = 1
        mapping = list(old_map)
        for dst in range(8):
            slots = [i for i, tile in enumerate(base) if tile // 64 == dst]
            for start in range(0, len(slots), window):
                group = slots[start:start + window]
                ordered = [old_map[i] for i in group]
                n_selected = sum(mask[base[j]] for j in ordered)
                front = [i for i in range(n_selected) if not mask[base[ordered[i]]]]
                tail = [i for i in range(n_selected, len(ordered)) if mask[base[ordered[i]]]]
                require(len(front) == len(tail), 'Codec placement exchange is not balanced')
                for early, late in zip(front, tail):
                    ordered[early], ordered[late] = ordered[late], ordered[early]
                for slot, old_slot in zip(group, ordered):
                    mapping[slot] = old_slot
        masks.append(mask)
        mappings.append(mapping)
    entry.update(mask=masks, maps=mappings, selected_per_source=list(map(sum, masks)))
    entry['coords'] = [[entry['base_coords'][rank][i] for i in mappings[rank]] for rank in range(8)]
    validate_entry(entry, {'tp': 8, 'window': window, 'fraction': count / 448}, entry['shape'])
    return entry


def record(entry, original):
    counts = list(map(sum, entry['mask']))
    moved = [sum(a != b for a, b in zip(entry['coords'][s], original['coords'][s])) for s in range(8)]
    return dict(entry=entry, entry_sha256=digest(entry), mask_sha256=mask_sha(entry['mask']),
                selected_per_source=counts, selected_total=sum(counts),
                selected_remote_fraction=sum(counts) / 3584,
                eligible_per_source=[448] * 8,
                wire_bytes_per_source=[14680064 - 15360 * q for q in counts],
                changed_vs_ma11_per_rank=moved,
                changed_vs_base_per_rank=[sum(i != j for i, j in enumerate(row)) for row in entry['maps']],
                mask_changed_vs_ma11_per_source=[sum(a != b for a, b in zip(row, original['mask'][s]))
                                                for s, row in enumerate(entry['mask'])])


def bundle(selection, counts=DEFAULT_COUNTS):
    counts = tuple(counts)
    require(counts == DEFAULT_COUNTS, 'Use the preregistered six-member MA20 bank')
    require(not selection.get('source_only') and selection.get('gpu_calibrated') is True,
            'MA20 requires the measured MA11 parent selection')
    roles = selection['variants'][C_POLICY]['roles']
    require(set(roles) == set(SHAPES), 'Both Attention and MLP plans are required')
    configs = {'MA11': dict(count=None, roles={})}
    configs.update({f'Q{count}': dict(count=count, roles={}) for count in counts})
    for role, shape in SHAPES.items():
        original = roles[role]['entry']
        require(original['shape'] == shape, 'Wrong MA11 role shape')
        window = selection['parent']['roles'][role]['config']['window']
        validate_entry(original, {'tp': 8, 'window': window, 'fraction': 1 / 64}, shape)
        require(roles[role]['mask_sha256'] == mask_sha(original['mask']), 'MA11 mask hash mismatch')
        configs['MA11']['roles'][role] = record(copy.deepcopy(original), original)
        for count in counts:
            configs[f'Q{count}']['roles'][role] = record(derive(original, count, window), original)
    return dict(protocol=PROTOCOL, baseline_selection=copy.deepcopy(selection),
                baseline_selection_sha256=digest(selection), counts=list(counts),
                configurations=configs, gpu_validated=False,
                runtime_calibration_required=True,
                mechanism='all_remote_eligible_then_codec_first_minimal_exchange',
                limitation='Logical submission positions are not CTA wave IDs; no latency benefit is inferred from this plan.')


def validate(plans):
    require(plans.get('protocol') == PROTOCOL, 'Wrong MA20 plan protocol')
    expected = bundle(plans['baseline_selection'], plans['counts'])
    require(plans == expected, 'MA20 mask, permutation, counts or provenance changed')
    return plans


def selection_for(plans, arm):
    """Return the existing worker selection schema with the requested real mask."""
    validate(plans)
    require(arm in plans['configurations'], 'Unknown MA20 configuration')
    result = copy.deepcopy(plans['baseline_selection'])
    if arm == 'MA11':
        return result
    for role, item in plans['configurations'][arm]['roles'].items():
        target = result['variants'][C_POLICY]['roles'][role]
        target.update(entry=copy.deepcopy(item['entry']), entry_sha256=item['entry_sha256'],
                      mask_sha256=item['mask_sha256'])
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('selection', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args(argv)
    require(not args.out.exists(), 'Do not overwrite a frozen MA20 plan')
    result = validate(bundle(read(args.selection)))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write(args.out, result)
    print(json.dumps({arm: {r: {k: v for k, v in rec.items() if k != 'entry'}
                           for r, rec in cfg['roles'].items()}
                      for arm, cfg in result['configurations'].items()}, indent=2))


if __name__ == '__main__':
    main()
