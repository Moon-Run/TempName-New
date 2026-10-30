"""Replay arrival ordering from returned calibration evidence, without GPU claims.

Only the plan and the raw inputs used here are verified. This is deliberately
not the executable-build admission path (returned archives may omit binaries).
Use ablation4.py fit on a fresh calibration to produce executable role plans.
"""
import argparse
import csv
import json
from pathlib import Path

from ablation_plan import resolve_input
from common import HERE, REPO, read, require, sha
import plan as fitter
from scope_plan import validate
from scope_protocol import SHAPES


def replay(selection_path, arrival_order='tail_residual', arrival_options=None):
    require(arrival_order in ('tail_residual', 'tail_assignment'), 'Unknown candidate order')
    require(arrival_options is None or arrival_order == 'tail_assignment', 'Margin options require tail_assignment')
    if arrival_order == 'tail_assignment':
        from arrival_assignment import normalize_options
        arrival_options = normalize_options(arrival_options)
    selection_path = Path(selection_path).resolve()
    selection = read(selection_path)
    parent = selection['parent']
    sources = [HERE/name for name in ('arrival_replay.py', 'plan.py', 'arrival_tail.py', 'common.py')]
    sources += [REPO/'tools/a800/arrival/plan.py', REPO/'tools/a800/phase5/plan.py']
    if arrival_order == 'tail_assignment':
        sources.append(HERE/'arrival_assignment.py')
    result = dict(diagnostic_only=True, gpu_validated=False, arrival_order=arrival_order,
                  algorithm_sources={str(path.relative_to(REPO)): sha(path) for path in sources},
                  selection_plan=dict(path=str(selection_path), sha256=sha(selection_path)),
                  roles={}, limitation='Raw-evidence CPU replay only; no executable plans or measured speedups. '
                  'BF16 labels and logical-slot proxies require independent same-codec GPU validation.')
    if arrival_options is not None:
        result['arrival_options'] = arrival_options
    for role in SHAPES:
        record = parent['roles'][role]
        validate(record, role, verify_evidence=False)
        require(record['origin']['kind'] == 'scope_bf16_calibration',
                'Replay requires the role-specific scope calibration')
        source = Path(record['origin']['source'])
        used = {}

        def evidence(name):
            key = str(source/name)
            require(key in record['inputs'], 'Unregistered calibration evidence: '+key)
            path = resolve_input(key, record['inputs'][key])
            used[key] = dict(resolved=str(path), sha256=record['inputs'][key])
            return path

        reports = [read(evidence(f'b0-interleaved-instrumented-rank{rank}.json')) for rank in range(8)]
        for rank, report in enumerate(reports):
            require(report['run_id'] == record['origin']['run_id'] and report['rank'] == rank and
                    report['world_size'] == 8 and report['policy'] == 'interleaved' and
                    report['completed'] and not report['smoke'], 'Calibration identity changed')
            require(all(report['hardware'][key] == record['hardware'][key] for key in ('host', 'devices')),
                    'Calibration hardware changed')
        with evidence(f'mapping-{role}.csv').open() as handle:
            rows = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        legacy = fitter.fit(reports, rows, record['config'], 'interleaved', SHAPES[role])
        # Establish that these exact observations reproduce the frozen physical
        # maps and mask before evaluating a new candidate on the same inputs.
        for field in ('maps', 'coords', 'base_coords', 'mask'):
            require(legacy[field] == record['entry'][field], 'Legacy replay differs: '+role+'/'+field)
        candidate = fitter.fit(reports, rows, record['config'], 'interleaved', SHAPES[role],
                               arrival_order=arrival_order, arrival_options=arrival_options)
        require(candidate['mask'] == legacy['mask'], 'Order replay changed quantization')
        diagnostics = candidate['arrival_order']
        changed_groups = [g for g in diagnostics['groups'] if g['status'] == 'candidate']
        result['roles'][role] = dict(inputs=used, legacy_changed_per_rank=legacy['changed_per_rank'],
            sampled_median_overhead_percent=legacy['sampled_median_overhead_percent'],
            legacy_heldout_near_critical_fraction=legacy['heldout_near_critical_fraction'],
            legacy_heldout_selected_observations=legacy['heldout_selected_observations'],
            candidate_changed_per_rank=candidate['changed_per_rank'], mask_unchanged=True,
            candidate_groups=len(changed_groups),
            heldout_candidate_groups=[dict(source=g['source'], destination=g['destination'],
                window_index=g['window_index'], **g['heldout']) for g in changed_groups],
            diagnostics=diagnostics)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('selection_plan', type=Path)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--arrival-order', choices=('tail_residual', 'tail_assignment'), default='tail_residual')
    parser.add_argument('--arrival-margin-multiplier', type=float,
                        help='Explicit v3 margin in [1,6]; absent preserves historical 6*error v2')
    args = parser.parse_args()
    if args.out:
        require(not args.out.exists(), 'Do not overwrite a replay result')
    options = None if args.arrival_margin_multiplier is None else dict(uncertainty_multiplier=args.arrival_margin_multiplier)
    result = replay(args.selection_plan, args.arrival_order, options)
    payload = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.out:
        with args.out.open('x') as handle:
            handle.write(payload)
        print(args.out)
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
