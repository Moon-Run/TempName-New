"""Audit one completed round when the user cancels the planned second round.

Keep frozen inputs unchanged and write a separate single-round verification.
This does not satisfy the repeated-round acceptance contract in assess.py.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys


def verify(root):
    def read(path):
        return json.loads((root / path).read_text())

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    sys.path.insert(0, str(root / 'scripts'))
    from routing import codec_placement, logical_bytes, module_routes, selection_entry

    info = read('submission.json')
    config = read('scripts/config.json')
    selection = read('scripts/selection-plan.json')
    protocol = read('results/model-1/protocol.json')
    preflight = read('results/preflight/preflight.json')
    analysis = read('results/model-1/analysis.json')
    assert info['repetitions'] == 2
    assert not list((root / 'results/model-2').glob('window-*'))
    assert preflight['completed'] and analysis['completed']
    assert protocol['script_sha256'] == preflight['script_sha256']
    assert protocol['flux_manifest_sha256'] == preflight['flux_manifest_sha256'] == info['build_manifest_sha256']
    for name, digest in info['files'].items():
        assert sha(root / name) == digest, name
    assert read('results/mapping/verification.json')['completed']
    for placement in info['codec_checks']:
        for rank in range(4):
            r = read(f'results/codec-{placement}/rank{rank}.json')
            assert r['completed'] and r['passed']
            if config.get('double_buffered') and placement.endswith('_arrival'):
                assert r['double_buffered'] and len(r['skewed_bursts']) == 2
                assert all(b['exact'] and b['calls'] == 12 for b in r['skewed_bursts'])
            if placement == 'native_taco':
                assert r['ste_gradient_gather_exact']
    build = Path(info['artifact_root']) / 'build'
    assert sha(build / 'manifest.json') == info['build_manifest_sha256']
    manifest = json.loads((build / 'manifest.json').read_text())
    frozen = Path(info['repo']) / 'outputs/a800/phase4/taco-fused-warp-20261003/taco/libflux_cuda.so'
    assert manifest['policies']['taco_fused']['library_sha256'] == sha(frozen)
    orders = protocol['plan']['orders']
    policies = info['policies']
    assert len(orders) == info['blocks'] == len(policies)
    for position in range(len(policies)):
        assert set(order[position] for order in orders) == set(policies)
    signatures, block_ms = {}, {p: {} for p in policies}
    windows = sorted((root / 'results/model-1').glob('window-*'))
    assert len(windows) == analysis['windows'] == len(orders) * len(policies)
    for index, window in enumerate(windows):
        assert (window / 'exit-code.txt').read_text().strip() == '0'
        rows = [json.loads((window / f'rank{rank}.json').read_text()) for rank in range(4)]
        for rank, r in enumerate(rows):
            assert r['passed'] and r['completed'] and r['skips'] == r['fallback_calls'] == 0
            assert r['policy'] == orders[index // len(policies)][index % len(policies)]
            assert r['block'] == index // len(policies) and r['window'] == index
            assert r['timing_protocol'] == 'full-step-one-clear-v2'
            assert r['warmup_steps'] == 10 and r['timed_steps'] == 20
            assert all(math.isfinite(v) for v in r['losses'] + r['grad_norms'])
            model = r['model']
            m, n = model['sequence'] * model['micro_batch'], model['hidden']
            assert r['module_policies'] == module_routes(r['observed_shapes'], r['policy'], model['layers'], model)
            if r['policy'] == 'native_taco':
                assert r['native_linear_forward_preserved'] and r['flux_library_loaded'] is False
            checks = r['selection_checks']
            if codec_placement(r['policy']) in ('fused', 'separate', 'tensor'):
                assert len(checks) == model['layers']
                mask = selection_entry(selection, r['policy'], m, n, model['ffn'])['mask'] if r['policy'].endswith('_selective') else None
                digest = hashlib.sha256(bytes(v for row in mask for v in row)).hexdigest() if mask is not None else None
                wire, bf16 = logical_bytes(mask, rank, m, n, model['tp'])
                assert all(name.endswith('.mlp.linear_fc2') and c['mask_sha256'] == digest and c['wire_bytes'] == wire and c['bf16_wire_bytes'] == bf16 for name, c in checks.items())
            else:
                assert not checks
            sig = tuple(r[k] for k in ('initial_parameters_sha256', 'initial_rng_sha256', 'tokens_sha256', 'gpu_uuid'))
            assert sig == signatures.setdefault(rank, sig)
        block_ms[rows[0]['policy']][rows[0]['block']] = max(r['ms_per_step'] for r in rows)
    for row in analysis['summary']:
        values = block_ms[row['policy']]
        assert math.isclose(row['median_ms_per_step'], statistics.median(values.values()), rel_tol=1e-12)
        for baseline in policies:
            reduction = 100 * (1 - math.exp(statistics.mean(math.log(t / block_ms[baseline][b]) for b, t in values.items())))
            assert math.isclose(row[f'latency_reduction_vs_{baseline}_percent'], reduction, abs_tol=1e-9)
    result = dict(completed_round_verified=True, planned_rounds=2, completed_rounds=[1],
                  second_round_started=False, repeat_confirmation_completed=False,
                  stop_reason='User explicitly requested ending after the first round.',
                  windows=len(windows), rank_records=4 * len(windows), global_timed_optimizer_steps=20 * len(windows),
                  single_round_position_balance_checked=True, within_round_initialization_match=True,
                  source_fingerprints_checked=True, required_flux_v2_fingerprint_checked=True,
                  mapping_codec_and_double_buffer_checks_passed=True, module_scope_and_logical_bytes_checked=True,
                  paired_point_estimates_recomputed=True, skipped_updates=0, fallback_calls=0,
                  scope='One balanced round in one allocation; no reversed repetition.',
                  verifier_sha256=sha(Path(__file__)))
    (root / 'verification-single-round.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    verify(parser.parse_args().root.resolve())
