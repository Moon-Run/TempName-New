"""Verify finished MLP-only quantization rounds, including cross-round initialization."""
import hashlib
import json
import math
from pathlib import Path
import sys
from routing import module_routes


def verify(root):
    info = json.loads((root/'submission.json').read_text())
    state = json.loads((root/'campaign-state.json').read_text())
    assert state['status'] == 'completed'
    for name, digest in info['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    for placement in ('fused','separate'):
        rows=[json.loads((root/f'results/codec-{placement}/rank{i}.json').read_text()) for i in range(4)]
        assert all(r['completed'] and r['passed'] for r in rows)
    protocols = [json.loads((root/f'results/model-{i}/protocol.json').read_text()) for i in (1, 2)]
    assert protocols[0]['script_sha256'] == protocols[1]['script_sha256']
    assert protocols[0]['flux_manifest_sha256'] == protocols[1]['flux_manifest_sha256']
    assert protocols[1]['plan']['orders'] == [list(reversed(o)) for o in reversed(protocols[0]['plan']['orders'])]
    signatures, records, windows = {}, 0, 0
    for i in (1, 2):
        folder = root/f'results/model-{i}'
        analysis = json.loads((folder/'analysis.json').read_text())
        expected = len(protocols[i-1]['plan']['orders'])*len(info['policies'])
        assert analysis['completed'] and analysis['windows'] == expected
        for window in sorted(folder.glob('window-*')):
            assert (window/'exit-code.txt').read_text().strip() == '0'
            windows += 1
            for rank in range(4):
                r = json.loads((window/f'rank{rank}.json').read_text())
                assert r['completed'] and r['passed'] and r['skips'] == r['fallback_calls'] == 0
                assert r['timing_protocol'] == 'full-step-one-clear-v2'
                assert r['warmup_steps'] == 10 and r['timed_steps'] == 20
                assert r['module_policies'] == module_routes(r['observed_shapes'], r['policy'])
                assert all(math.isfinite(v) for v in r['losses']+r['grad_norms'])
                sig = tuple(r[k] for k in ('initial_parameters_sha256','initial_rng_sha256','tokens_sha256','gpu_uuid'))
                assert sig == signatures.setdefault(rank, sig), (i, window, rank)
                records += 1
    assert records == 4*windows and windows == info['repetitions']*info['blocks']*len(info['policies'])
    result = dict(completed=True, windows=windows, rank_records=records,
                  global_timed_optimizer_steps=20*windows, timed_steps_per_policy=20*info['repetitions']*info['blocks'],
                  original_mapping_builds=True, module_scope_checked=True,
                  cross_round_initialization_match=True, reverse_order_checked=True,
                  skipped_updates=0, fallback_calls=0, finite_losses_and_gradients=True,
                  scope='Two process repetitions in one allocation; not independent jobs',
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (root/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
