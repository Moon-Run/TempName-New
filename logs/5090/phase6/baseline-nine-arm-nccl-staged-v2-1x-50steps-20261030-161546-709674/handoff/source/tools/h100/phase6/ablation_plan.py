"""CPU-only, explicit native/G0–G6 definitions and physical-tile derivation."""
import copy
import hashlib
import math
from pathlib import Path
import random

from common import HERE, REPO, PROTOCOL as PARENT_PROTOCOL, read, require, sha, validate_entry

PROTOCOL = 'h100-gpt67-tp8-ablation-v1'
FREEZE_ID = 'h100-phase6-candidate-v1-20261013'
FREEZE = REPO / 'outputs/h100/phase6/frozen-candidate-v1-20261013'
BASES = ('remote_first', 'interleaved')
GROUPS = {
    'native': ('native', False, 'native_bf16'),
    'G0': ('original', False, 'bf16'),
    'G1': ('base', False, 'bf16'),
    'G2': ('base', True, 'bf16'),
    'G3': ('base', False, 'targeted'),
    'G4': ('base', True, 'targeted'),
    'G5': ('base', True, 'random_matched'),
    'G6': ('base', True, 'all_remote'),
}


def experiment(protocol):
    """Select an explicit protocol without changing the legacy defaults."""
    import sys
    if protocol == PROTOCOL:
        return sys.modules[__name__]
    import ablation2_plan
    require(protocol == ablation2_plan.PROTOCOL, 'Unknown ablation protocol')
    return ablation2_plan


def definitions(base, seeds=(17, 29, 43)):
    require(base in BASES, 'Only remote_first and interleaved are in scope')
    require(len(seeds) >= 3 and len(set(seeds)) == len(seeds) and
            all(type(s) is int and 0 <= s < 2**32 for s in seeds),
            'Register at least three distinct uint32 G5 seeds before measuring')
    result = {}
    for group, (order, arrival, mode) in GROUPS.items():
        for seed in seeds if group == 'G5' else (None,):
            key = f'G5-s{seed}' if group == 'G5' else group
            result[key] = dict(group=group, base=base, order=base if order == 'base' else order,
                               arrival=arrival, quantization_mode=mode, seed=seed,
                               implementation='native_megatron' if group == 'native' else
                               'frozen-candidate-v1-sm90-gemmv2',
                               parent_freeze=FREEZE_ID)
    return result


def _input_paths(name):
    """Original/container/local locations for the same registered input."""
    path = Path(name)
    candidates = [path]
    prefix = '/workspace/TempName/'
    if str(path).startswith(prefix):
        relative = Path(str(path)[len(prefix):])
    else:
        try:
            relative = path.relative_to(REPO)
        except ValueError:
            relative = None
    if relative is not None:
        candidates += [REPO / relative, FREEZE / 'source' / relative]
    return candidates, relative


def resolve_input(name, digest, registered_inputs=None):
    """Resolve exact bytes, including controller snapshots sealed by this plan.

    A historical calibration may register both its original tools/ path and
    the copied controllers/ path. Editing the live tools must not invalidate
    that immutable evidence. Never use unregistered copies or substitute raw
    measurements by basename; all candidates still need the expected SHA256.
    """
    candidates, relative = _input_paths(name)
    for candidate in candidates:
        if candidate.is_file() and sha(candidate) == digest:
            return candidate.resolve()
    if (registered_inputs and relative is not None and relative.parent == Path('tools/h100/phase6')
            and relative.suffix == '.py' and registered_inputs.get(str(name)) == digest):
        for archived, expected in registered_inputs.items():
            snapshot = Path(archived)
            if expected == digest and snapshot.parent.name == 'controllers' and snapshot.name == relative.name:
                paths, _ = _input_paths(archived)
                for candidate in paths:
                    if candidate.is_file() and sha(candidate) == digest:
                        return candidate.resolve()
    raise ValueError(f'Missing/changed frozen input: {name} ({digest})')


def parent_plan(path=None):
    freeze = read(FREEZE / 'manifest.json')
    require(freeze['freeze_id'] == FREEZE_ID, 'Wrong candidate freeze')
    path = Path(path) if path else FREEZE / 'evidence/selection-plan.json'
    p = read(path)
    require(p.get('protocol') == PARENT_PROTOCOL and p.get('arch') == 'sm90' and
            p.get('gpu_calibrated') is True and not p.get('test_only'), 'Real H100 calibration required')
    require(p['config'] == freeze['config'] and p['world'] == 8, 'Only frozen GPT 6.7B TP8 configuration is supported')
    require(set(p['policies']) == set(BASES) and p['training_passes'] == [0, 1] and
            p['heldout_passes'] == [2], 'Independent bases; passes 0/1 train, pass 2 diagnostic only')
    if path.resolve() == (FREEZE / 'evidence/selection-plan.json').resolve():
        require(sha(path) == freeze['candidate']['plan_sha256'], 'Frozen candidate plan changed')
    require(p['inputs'], 'Calibration evidence is required')
    evidence = {name: dict(resolved=str(resolve_input(name, digest)), sha256=digest)
                for name, digest in p['inputs'].items()}
    for base in BASES:
        require(len(p['policies'][base]) == 1, 'One exact MLP shape required')
        validate_entry(p['policies'][base][0], p['config'])
    devices = p['hardware']['devices']
    require(len(devices) == 8 and len({d['uuid'] for d in devices}) == 8 and
            all('H100' in d['name'] and d['sm_count'] == 132 and d['capability'] == [9, 0]
                for d in devices), 'H100 TP8/132 SM calibration required')
    return p, path.resolve(), evidence


def counts(mask):
    world = len(mask)
    per = len(mask[0]) // world
    return [[sum(row[dst*per:(dst+1)*per]) for dst in range(world)] for row in mask]


def mask_sha(mask):
    return hashlib.sha256(bytes(v for row in mask for v in row)).hexdigest()


def derive(parent, definition):
    require(definition['group'] not in ('native', 'native_taco'), 'Native has no tile plan')
    c = parent['config']
    source = parent['policies'][definition['base']][0]
    e = {k: copy.deepcopy(source[k]) for k in
         ('shape', 'M', 'N', 'K_global', 'K_local', 'tiles', 'maps', 'coords', 'base_coords', 'mask')}
    world, tiles = c['tp'], e['tiles']
    per = tiles // world
    if definition['order'] == 'original':
        # Undo only rank+1 -> rank rotation of the measured StreamK mapping.
        # Do not assume a row-major StreamK raster or infer it from interleaved.
        remote = parent['policies']['remote_first'][0]['base_coords']
        e['base_coords'] = [[(t-per) % tiles for t in row] for row in remote]
    if not definition['arrival']:
        e['maps'] = [list(range(tiles)) for _ in range(world)]
    e['coords'] = [[e['base_coords'][r][i] for i in e['maps'][r]] for r in range(world)]
    mode = definition['quantization_mode']
    if mode == 'bf16':
        e['mask'] = [[0]*tiles for _ in range(world)]
    elif mode == 'all_remote':
        e['mask'] = [[int(t//per != r) for t in range(tiles)] for r in range(world)]
    elif mode == 'random_matched':
        rng = random.Random(definition['seed'])
        e['mask'] = [[0]*tiles for _ in range(world)]
        target = counts(source['mask'])
        for r in range(world):
            for dst in range(world):
                for t in rng.sample(list(range(dst*per, (dst+1)*per)), target[r][dst]):
                    e['mask'][r][t] = 1
        e['random_choices_per_partition'] = [[math.comb(per, n) for n in row] for row in target]
    else:
        require(mode == 'targeted', 'Unknown explicit quantization mode')
    e.update(selected_per_source=list(map(sum, e['mask'])),
             selected_per_destination=counts(e['mask']), mask_sha256=mask_sha(e['mask']),
             changed_per_rank=[sum(i != j for i, j in enumerate(row)) for row in e['maps']])
    validate(e, c, definition, source)
    return e


def validate(e, c, d, source):
    # Structural validation is independent of the selection budget. G6 explicitly
    # enumerates all remote tiles; it never calls the scored fraction selector.
    world, tiles = c['tp'], e['tiles']
    per = tiles // world
    require(e['shape'] == source['shape'] and e['K_local'] == source['K_local'], 'Shape changed')
    for field in ('maps', 'coords', 'base_coords', 'mask'):
        require(len(e[field]) == world and all(len(row) == tiles for row in e[field]), 'Bad plan dimensions')
    for rank in range(world):
        maps, coords, base, mask = (e[f][rank] for f in ('maps', 'coords', 'base_coords', 'mask'))
        require(all(sorted(row) == list(range(tiles)) for row in (maps, coords, base)), 'Non-bijective plan')
        require(coords == [base[i] for i in maps], 'Logical/physical mapping mismatch')
        positions = {}
        for dst in range(world):
            slots = [i for i, tile in enumerate(base) if tile//per == dst]
            positions.update({i: (dst, j//c['window']) for j, i in enumerate(slots)})
        require(all(positions[i] == positions[j] for i, j in enumerate(maps)), 'Arrival crosses destination/window')
        require(all(type(v) is int and v in (0, 1) for v in mask) and
                not any(mask[rank*per:(rank+1)*per]), 'Invalid mask/local quantization')
        mode = d['quantization_mode']
        if mode == 'bf16':
            require(not any(mask), 'BF16 must bypass every encode/decode contribution')
        elif mode == 'all_remote':
            require(mask == [int(t//per != rank) for t in range(tiles)], 'G6 must select every remote contribution')
        else:
            require(sum(mask) <= int((tiles-per)*c['fraction']), 'Selective budget exceeded')
        if not d['arrival']:
            require(maps == list(range(tiles)), 'Arrival disabled but permutation remains')
        else:
            require(maps == source['maps'][rank], 'Arrival table changed')
    if d['quantization_mode'] == 'targeted':
        require(e['mask'] == source['mask'], 'Targeted physical mask changed')
    if d['quantization_mode'] == 'random_matched':
        require(counts(e['mask']) == counts(source['mask']), 'G5 must match actual source/destination counts')


def schedule(base, definitions_, blocks=10, seed=20261013):
    require(type(blocks) is int and blocks >= 2, 'At least two paired blocks per round')
    rng = random.Random(seed)
    first = []
    for _ in range(blocks):
        row = list(definitions_)
        rng.shuffle(row)
        first.append(row)
    result = []
    for rep, rows in enumerate((first, [list(reversed(r)) for r in reversed(first)]), 1):
        index = 0
        for block, row in enumerate(rows):
            for key in row:
                result.append(dict(base=base, round=rep, block=block, index=index, policy=key,
                                   directory=f'{base}/round{rep}/window-{index:03d}-{key}'))
                index += 1
    return result
