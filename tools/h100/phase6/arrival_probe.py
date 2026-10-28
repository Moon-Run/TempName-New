"""Read-only admission gate for an already measured arrival-order candidate.

This is a CPU audit of ablation4/7/7-new full optimizer-step evidence, not a GPU
benchmark. Default inspection validates frozen plans/build snapshots and rejects
identity orders before timing work. --evaluate replays the existing strict
preflight/report in a temporary directory, leaving the experiment untouched.
Archived reports must match the current strict schema except for offline mode.

Only one preselected B versus A is admitted. Two distinct run_ids with identical
plans, implementations, masks, recorded binaries and measurement identities are
required; the complete-block B/A CI must be positive in every round and run.
This gate cannot establish per-role speedups, tune on pass 2, or deploy a kernel.
"""
import argparse
import hashlib
import importlib
import json
import math
from pathlib import Path
import tempfile

from common import read, require, sha


PROTOCOL = 'h100-arrival-evidence-gate-v1'
ORDER_FILES = {'src/gemm_rs/tile_scheduler/threadblock_swizzle.hpp', 'phase6_host.hpp'}
OUTPUTS = {'preflight.json', 'report.json', 'report.md'}
REPORTERS = {
    'h100-gpt67-tp8-ablation4-v1': 'ablation4_report',
    'h100-gpt67-tp8-ablation7-v1': 'ablation7_report',
    'h100-gpt67-tp8-ablation7-new-v1': 'ablation7_new_report',
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_same_json(actual, expected, message):
    if digest(actual) != digest(expected):
        keys = sorted(k for k in actual.keys() | expected.keys()
                      if k not in actual or k not in expected or digest(actual[k]) != digest(expected[k]))
        raise ValueError(message+'; changed fields: '+', '.join(keys))


def pair_contract(routing, plans, sources):
    """Check the schedule-only contrast after the protocol verifier ran."""
    a, b = ((routing.A, routing.B) if hasattr(routing, 'A') else routing.POLICIES[2:4])
    definitions = [routing.definition(p) for p in (a, b)]
    order_keys = {'policy', 'family', 'order', 'arrival', 'arrival_method'}
    require({k: v for k, v in definitions[0].items() if k not in order_keys} ==
            {k: v for k, v in definitions[1].items() if k not in order_keys},
            'A/B execution, codec or lifecycle differs')
    common = [{k: v for k, v in source.items() if k not in ORDER_FILES} for source in sources]
    require(common[0] and common[0] == common[1], 'A/B compute/codec/wrapper source hashes differ')
    roles = {}; fingerprints = {}
    aa, bb = (routing.role_plans(plans, p) for p in (a, b))
    require(set(aa) == set(bb) == {'attn', 'mlp'}, 'Both role plans are required')
    for role in aa:
        x, y = aa[role]['entry'], bb[role]['entry']
        require({k: v for k, v in x.items() if k not in ('maps', 'coords')} ==
                {k: v for k, v in y.items() if k not in ('maps', 'coords')},
                'A/B shape, mask or base mapping differs: '+role)
        require(len(x['coords']) == len(y['coords']) == 8 and
                all(len(row) == 512 for row in x['coords']+y['coords']), 'Incomplete TP8 mapping')
        changed = [sum(i != j for i, j in zip(ar, br)) for ar, br in zip(x['coords'], y['coords'])]
        roles[role] = dict(changed_per_rank=changed, identical_to_A=not any(changed))
        fingerprints[role] = dict(A_entry_sha256=aa[role]['entry_sha256'],
                                  B_entry_sha256=bb[role]['entry_sha256'],
                                  mask_sha256=aa[role]['mask_sha256'])
    return dict(A=a, B=b, roles=roles, identity=all(r['identical_to_A'] for r in roles.values()),
                fingerprints=fingerprints, sources=sources)


def inspect(root):
    root = Path(root).resolve()
    protocol = read(root/'submission.json')['protocol']
    require(protocol in REPORTERS, 'Only strict ablation4/7/7-new evidence is supported')
    reporter = importlib.import_module(REPORTERS[protocol]); routing = reporter.routing
    # Offline validates returned source/build/plan seals and recorded library
    # identities. It deliberately does not claim to reload/re-hash remote .so's.
    info = reporter.verify_inputs(root, offline=True)
    require(info['executable'] and not info['source_only'], 'CPU fixtures cannot certify GPU evidence')
    plans = read(root/'scripts/selection-plan.json')
    if protocol.endswith(('ablation7-v1', 'ablation7-new-v1')):
        fused = read(root/'provenance/parent-fused-manifest.json')
        improved = read(root/'provenance/improved-manifest.json')
        sources = [fused['variants']['interleaved_all']['source_sha256'],
                   improved['variants'][reporter.builder.IMPROVED_NAME]['source_sha256']]
    else:
        fused = read(root/'provenance/fused-manifest.json')
        sources = [fused['variants'][p]['source_sha256'] for p in ('interleaved_all', 'interleaved_arrival_all')]
    pair = pair_contract(routing, plans, sources)
    config = read(root/'scripts/config.json')
    signature = dict(protocol=protocol, model=info['config'], communication=info['communication'],
                     timing={k: config[k] for k in ('warmup_steps', 'timed_steps')},
                     hardware=info['calibration_hardware'], pair=pair,
                     libraries={p: {name: lib['sha256'] for name, lib in info['variants'][p]['libraries'].items()}
                                for p in (pair['A'], pair['B'])})
    if 'arrival_options' in plans:
        # Identical maps alone cannot join runs selected with different gates.
        signature.update(arrival_options=plans['arrival_options'],
                         selection_plan_sha256=sha(root/'scripts/selection-plan.json'))
    return dict(root=root, reporter=reporter, info=info, pair=pair, signature=signature,
                signature_sha256=digest(signature))


def replay_report(record):
    """Recompute every preflight/timing check without overwriting source files."""
    root, reporter = record['root'], record['reporter']
    saved_gate, saved_report = read(root/'preflight.json'), read(root/'report.json')
    with tempfile.TemporaryDirectory(prefix='arrival-evidence-audit-') as folder:
        audit = Path(folder)
        for entry in root.iterdir():
            if entry.name not in OUTPUTS:
                (audit/entry.name).symlink_to(entry, target_is_directory=entry.is_dir())
        gate = reporter.preflight(audit, offline=True)
        # Reporters return some tuples, while JSON stores those as arrays.
        require_same_json(gate, saved_gate, 'Recomputed preflight differs from archived evidence')
        # The report seals the preflight FILE, not just its JSON content. The
        # returned archive may preserve another dictionary order/whitespace.
        # Restore the now independently verified original bytes in our scratch
        # tree so this link in the evidence chain keeps its original SHA256.
        (audit/'preflight.json').write_bytes((root/'preflight.json').read_bytes())
        result = reporter.report(audit, offline=True)
        normalize = lambda value: {k: v for k, v in value.items() if k != 'offline'}
        require_same_json(normalize(result), normalize(saved_report),
                          'Recomputed report differs from archived evidence')
    require(result['completed'] and result['correctness_passed'], 'Incomplete or incorrect GPU evidence')
    return result


def paired_gate(result, a, b):
    """All samples and all rounds are mandatory; never choose a favorable subset."""
    expected = {str(i) for i in range(1, result['round_count']+1)} | {'combined'}
    require(set(result['rounds']) == expected, 'Missing or extra timing round')
    comparisons = {}; positive = result['round_count'] >= 2
    for name, summary in result['rounds'].items():
        comparison = summary['comparisons'][b+'/'+a]
        ci = comparison['ci95']
        require(ci is None or (len(ci) == 2 and all(math.isfinite(v) for v in ci) and ci[0] <= ci[1]),
                'Invalid paired confidence interval')
        require(math.isfinite(comparison['percent']), 'Non-finite paired effect')
        passed = summary['blocks'] >= 2 and ci is not None and ci[0] > 0
        positive = positive and passed
        comparisons[name] = dict(blocks=summary['blocks'], percent=comparison['percent'], ci95=ci, passed=passed)
    return dict(passed=positive, comparisons=comparisons,
                rule='Two reversed rounds; every round and combined B/A complete-block 95% CI lower bound > 0')


def probe(roots, evaluate=False):
    require(bool(roots), 'At least one compare directory is required')
    records = [inspect(root) for root in roots]
    ids = [r['info']['run_id'] for r in records]
    require(len(ids) == len(set(ids)), 'Duplicate run_id cannot count as independent confirmation')
    first = records[0]
    require(all(r['signature'] == first['signature'] for r in records),
            'Comparisons must use identical plans, masks, source, binaries, hardware and configuration')
    result = dict(protocol=PROTOCOL, status='dry_run_needs_evaluation', validated_for_same_configuration=False,
                  evaluated=False, gpu_launched=False, scope='joint_attn_mlp_full_optimizer_step',
                  execution_timing=first['signature']['timing'],
                  configuration_sha256=first['signature_sha256'], roles=first['pair']['roles'],
                  distinct_run_ids=ids, minimum_independent_runs=2, runs=[],
                  limitation='Read-only offline audit of recorded GPU evidence; does not reload remote libraries, '
                             'validate individual roles, prove transfer to another configuration, or deploy a kernel. '
                             'Pointwise confidence intervals do not correct repeated candidate selection.')
    for record in records:
        result['runs'].append(dict(root=str(record['root']), run_id=record['info']['run_id'],
                                   submission_sha256=sha(record['root']/'submission.json'),
                                   plans_sha256=sha(record['root']/'scripts/selection-plan.json')))
    if first['pair']['identity']:
        result['status'] = 'not_tested_identity'
        result['reason'] = 'A/B use the same physical mapping in both roles; no timing bootstrap or new GPU run is needed.'
        return result
    if not evaluate:
        return result
    result['evaluated'] = True
    measurement = None
    for record, run in zip(records, result['runs']):
        report = replay_report(record)
        if measurement is None:
            measurement = report['measurement_identity']
        require(report['measurement_identity'] == measurement,
                'Independent comparisons changed worker/software/RNG/tokens or recorded libraries')
        run.update(paired=paired_gate(report, record['pair']['A'], record['pair']['B']),
                   execution_timing=report['execution_timing'],
                   report_sha256=sha(record['root']/'report.json'),
                   preflight_sha256=sha(record['root']/'preflight.json'))
    result['measurement_identity_sha256'] = digest(measurement)
    if len(records) < 2:
        result['status'] = 'inconclusive_need_confirmation'
    elif not all(run['paired']['passed'] for run in result['runs']):
        result['status'] = 'not_validated_no_repeatable_gain'
    else:
        result.update(status='validated_for_same_configuration', validated_for_same_configuration=True)
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('compare', type=Path, nargs='+', help='Existing ablation4/7/7-new compare directories')
    parser.add_argument('--evaluate', action='store_true', help='Replay archived preflight/report; CPU only, no GPU launch')
    parser.add_argument('--out', type=Path, help='New JSON outside all evidence directories; default stdout')
    args = parser.parse_args()
    if args.out:
        output = args.out.resolve()
        require(not output.exists(), 'Do not overwrite an existing output')
        require(all(root.resolve() not in output.parents for root in args.compare),
                'Output must be outside every original evidence directory')
    result = probe(args.compare, args.evaluate)
    encoded = json.dumps(result, indent=2, ensure_ascii=False)+'\n'
    if args.out:
        with args.out.open('x') as stream:
            stream.write(encoded)
    else:
        print(encoded, end='')


if __name__ == '__main__':
    main()
