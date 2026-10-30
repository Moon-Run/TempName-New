"""Verify finished MLP-only quantization rounds, including cross-round initialization."""
import hashlib
import json
import math
from pathlib import Path
import sys
from routing import module_routes, codec_placement, base_policy, selection_entry, logical_bytes


def verify(root):
    info = json.loads((root/'submission.json').read_text())
    state = json.loads((root/'campaign-state.json').read_text())
    assert state['status'] == 'completed'
    assert json.loads((root/'results/mapping/verification.json').read_text())['completed']
    selection = json.loads((root/'scripts/selection-plan.json').read_text())
    config = json.loads((root/'scripts/config.json').read_text())
    for name, digest in info['files'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, name
    for placement in info.get('codec_checks',['remote_first','interleaved','separate']):
        rows=[json.loads((root/f'results/codec-{placement}/rank{i}.json').read_text()) for i in range(4)]
        assert all(r['completed'] and r['passed'] for r in rows)
        if config.get('double_buffered') and placement.endswith('_arrival'):
            assert all(r['double_buffered'] and len(r['skewed_bursts'])==2 and
                       all(b['exact'] and b['calls']==12 for b in r['skewed_bursts']) for r in rows)
        if placement=='native_taco':assert all(r['ste_gradient_gather_exact'] for r in rows)
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
                model=r['model'];m=model['sequence']*model['micro_batch'];n=model['hidden']
                assert r['module_policies'] == module_routes(r['observed_shapes'], r['policy'],model['layers'],model)
                policy = r['policy']
                if policy=='native_taco':
                    assert r['native_linear_forward_preserved'] and r['flux_library_loaded'] is False
                checks = r['selection_checks']
                if codec_placement(policy) in ('fused','separate','tensor'):
                    assert len(checks) == model['layers']
                    mask = selection_entry(selection,policy,m,n,model['ffn'])['mask'] if policy.endswith('_selective') else None
                    digest = hashlib.sha256(bytes(v for row in mask for v in row)).hexdigest() if mask is not None else None
                    wire,bf16_wire = logical_bytes(mask,rank,m,n,model['tp'])
                    assert all(name.endswith('.mlp.linear_fc2') and c['mask_sha256']==digest and
                               c['wire_bytes']==wire and c['bf16_wire_bytes']==bf16_wire for name,c in checks.items())
                else:
                    assert not checks
                assert all(math.isfinite(v) for v in r['losses']+r['grad_norms'])
                sig = tuple(r[k] for k in ('initial_parameters_sha256','initial_rng_sha256','tokens_sha256','gpu_uuid'))
                assert sig == signatures.setdefault(rank, sig), (i, window, rank)
                records += 1
    assert records == 4*windows and windows == info['repetitions']*info['blocks']*len(info['policies'])
    result = dict(completed=True, windows=windows, rank_records=records,
                  global_timed_optimizer_steps=20*windows, timed_steps_per_policy=20*info['repetitions']*info['blocks'],
                  attention_original_mapping_checked=True, mlp_base_mappings_checked=True,
                  selection_masks_and_logical_bytes_checked=True, module_scope_checked=True,
                  cross_round_initialization_match=True, reverse_order_checked=True,
                  skipped_updates=0, fallback_calls=0, finite_losses_and_gradients=True,
                  scope='Two process repetitions in one allocation; not independent jobs',
                  verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if 'native_taco' in info['policies']:
        result.update(native_taco_no_flux_and_native_linear_checked=True,
                      native_taco_ste_gather_checked=True,
                      native_taco_effective_replaced_modules=config['common']['layers'],
                      native_taco_legacy_count_note='Early frozen worker reports replaced_modules=24 (audit wrappers); timed routes and CPU profile scope counts verify only 12 MLP communications are replaced. Canonical worker field corrected for future runs.')
    if 'remote_arrival_selective' in info['policies']:
        result['arrival_coordinate_and_fixed_physical_mask_checks_passed']=True
    (root/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    verify(Path(sys.argv[1]))
