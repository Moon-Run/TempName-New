"""Portable three-arm TP8/DP2 experiment contracts; no Torch import."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics

PROTOCOL = 'h100-tp8-dp2-three-arm-v1'
ARMS = ('ma11', 'flux-basic', 'megatron-cuda')
LABELS = dict(zip(ARMS, ('MA11', 'Flux＋基础量化 v2', 'Megatron＋CUDA量化')))
POLICIES = {'ma11': 'interleaved_arrival_selective_attn_mlp', 'flux-basic': 'basic',
            'megatron-cuda': 'native_cuda_taco_attn_mlp'}
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
             micro_batch=1, global_batch=4, tp=8, dp=2, sm_count=132,
             seed=1234, token_seed=4321, vocab=32768)
TP, DP, WORLD, MICROBATCHES = 8, 2, 16, 2
MEGATRON_COMMIT = '3ea68ad6042cc1204386ae9364358f7c4de1bc37'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def build_inputs(info):
    """Only inputs consumed by the unchanged native build recipe."""
    def selected(name):
        return name.startswith(('native/', 'deps/')) or name == 'app/tp8_dp2_build.py'
    require('app/tp8_dp2_build.py' in info['files'] and
            all(any(n.startswith(prefix) for n in info['files']) for prefix in ('native/', 'deps/')),
            'Build input manifest is incomplete')
    return dict(files={k:v for k,v in info['files'].items() if selected(k)},
                symlinks={k:v for k,v in info['symlinks'].items() if selected(k)},
                model=info['model'], arms=info['arms'], dependencies=info.get('dependencies'))


def build_package_id(root, info):
    """Reuse a prior build only with its full original manifest and exact inputs.

    Worker/results identity still uses the new package ID. This does not allow
    partial timing runs to resume, or caches from changed CUDA sources/recipes.
    """
    reference=info.get('build_reference')
    if reference is None:
        return info['package_id']
    require(reference in info['files'] and sha(Path(root)/reference)==info['files'][reference],
            'Original build package manifest changed')
    original=read(Path(root)/reference)
    require(original['package_id']==digest({k:v for k,v in original.items() if k!='package_id'}) and
            original['protocol']==info['protocol'], 'Invalid original build package identity')
    require(build_inputs(original)==build_inputs(info), 'Native build inputs differ from the referenced package')
    return original['package_id']


def verify_package(root):
    root = Path(root)
    info = read(root/'package.json')
    require(info['protocol'] == PROTOCOL and info['model'] == MODEL and info['arms'] == list(ARMS),
            'Wrong TP8/DP2 package identity')
    require(info['package_id'] == digest({k:v for k,v in info.items() if k != 'package_id'}), 'Package manifest identity changed')
    for name, value in info['files'].items():
        require(sha(root/name) == value, 'Package file changed: '+name)
    for name, target in info['symlinks'].items():
        import os
        require((root/name).is_symlink() and os.readlink(root/name) == target,
                'Package dependency link changed: '+name)
        require((root/name).resolve().is_relative_to(root.resolve()), 'Package symlink escapes its root')
    build_package_id(root,info)
    return info


def schedule(trials=1, blocks=12):
    require(trials in (1, 2) and blocks >= 6 and blocks % 6 == 0, 'Use 1/2 trials and a positive multiple of six blocks')
    orders = list(itertools.permutations(ARMS))
    random.Random(20261028).shuffle(orders)
    return [dict(trial=t+1, block=b, index=b*3+i, arm=arm, mode='timing',
                 directory=f'compare{t+1}/block{b:02d}-{arm}')
            for t in range(trials) for b in range(blocks)
            for i, arm in enumerate(orders[b % 6])]


def jobs(trials=1, blocks=12):
    return ([dict(trial=0, block=-1, index=i, arm=a, mode='smoke', directory='smoke-'+a)
             for i,a in enumerate(ARMS)] +
            [dict(trial=0, block=-1, index=i, arm=a, mode='profile', directory='profile-'+a)
             for i,a in enumerate(ARMS)] + schedule(trials, blocks))


def validate_projection_checks(row, mode):
    require(mode in ('smoke','timing','profile'), 'Unknown projection validation mode')
    checks=row['forward_checks']
    if mode=='smoke':
        require(isinstance(checks,dict) and len(checks)==64 and
                set(checks)==set(row['target_modules']) and
                all(v.get('passed') is True for v in checks.values()),
                'Incomplete original projection numeric checks')
    else:
        # The frozen adapters deliberately run reference GEMMs/collectives only
        # during smoke. Their absence is required in timing and profile workers.
        require(checks=={}, 'Projection numeric audit leaked into timing/profile')


def validate_rows(rows, job, package_id):
    require(len(rows) == WORLD and {r['rank'] for r in rows} == set(range(WORLD)), 'Missing/duplicate global rank records')
    rows = sorted(rows, key=lambda r:r['rank'])
    warmup, count = (2, 3) if job['mode'] == 'smoke' else (10, 1 if job['mode'] == 'profile' else 50)
    calls = (warmup+count)*MICROBATCHES
    for rank,row in enumerate(rows):
        require(row['completed'] and row['passed'] and row['skips'] == 0 and
                row['protocol'] == PROTOCOL and row['package_id'] == package_id and
                row['arm'] == job['arm'] and row['mode'] == job['mode'] and row['block'] == job['block'] and
                row['window'] == job['index'] and row['world_size'] == WORLD and
                row['local_rank'] == row['tp_rank'] == rank % TP and row['dp_rank'] == rank//TP and
                row['tp_group_ranks'] == list(range(rank//TP*TP, (rank//TP+1)*TP)) and
                row['dp_group_ranks'] == [rank % TP, rank % TP+TP], 'Worker identity or TP/DP mapping changed')
        require(row['model']['global_batch'] == 4 and row['microsteps'] == 2 and
                row['sample_ids'] == list(range(rank//TP*2, rank//TP*2+2)) and
                row['tokens_per_optimizer_step'] == 8192 and row['warmup_steps'] == warmup and
                len(row['losses']) == len(row['grad_norms']) == count and
                all(math.isfinite(float(v)) for v in row['losses']+row['grad_norms']), 'Wrong workload or non-finite training')
        require(len(row['target_modules']) == 64 and set(row['target_modules']) == set(row['timed_calls']) and
                all(v == count*MICROBATCHES for v in row['timed_calls'].values()) and
                row['fallback_calls'] == 0, 'Missing projection calls or fallback')
        validate_projection_checks(row,job['mode'])
        for key in ('peak_allocated_gib','peak_reserved_gib'):
            require(math.isfinite(row[key]) and row[key] > 0, 'Missing peak GPU memory')
        if job['mode'] != 'profile':
            require(row['timed_steps'] == count and math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] > 0,
                    'Missing wall-clock timing')
        if job['arm'] == 'ma11':
            cache,zero,softmax = (row['ma11_activation_setup'], row['ma11_zero_count_state'], row['ma11_softmax_state'])
            require(cache['installed'] and len(cache['module_roles']) == 64 and
                    zero['installed'] and zero['failed'] is False and zero['calls'] == warmup+count and
                    zero['kernel_calls'] == 2*(warmup+count) and zero['initializations'] == 1 and
                    zero['selected_tensors'] == (389 if rank % TP == 0 else 194), 'MA8/MA9 installation or counters changed')
            require(zero['audits'] == (warmup+count if job['mode'] == 'smoke' else 0), 'MA9 audit boundary changed')
            require(softmax['installed'] and len(softmax['modules']) == 32 and
                    softmax['shape'] == [1,4,2048,2048] and
                    softmax['forward_audits'] == softmax['backward_audits'] ==
                    (calls*32 if job['mode']=='smoke' else 0), 'MA11 softmax missing or audit boundary changed')
            require((softmax['self_test'] is not None and softmax['self_test']['passed'] and
                     zero['self_test'] is not None and zero['self_test']['passed']) if job['mode']=='smoke'
                    else softmax['self_test'] is zero['self_test'] is None, 'Native arithmetic self-test boundary changed')
            require(len(row['ma11_consumer_state']) == 64 and all(
                v['projection_calls'] == v['bda_calls'] == calls for v in row['ma11_consumer_state'].values()),
                'CR3 fused consumer missing')
        else:
            require(len(row['codec_calls']) == 64 and all(v == calls for v in row['codec_calls'].values()),
                    'Standalone CUDA codec coverage changed')
            if job['mode'] == 'smoke':
                require(row['codec_validation']['passed'] is True and row['codec_validation']['synthetic'] is False,
                        'Codec GPU preflight missing')
            else:
                require(row['codec_validation'] is None, 'Codec audit leaked into timing/profile')
            if job['arm'] == 'megatron-cuda':
                require(row['native_linear_forward_preserved'] and row['flux_library_loaded'] is False,
                        'Megatron CUDA baseline changed GEMM path')
    require(len({r['gpu_uuid'] for r in rows}) == WORLD and
            len({r['hardware']['host'] for r in rows[:TP]}) == len({r['hardware']['host'] for r in rows[TP:]}) == 1 and
            rows[0]['hardware']['host'] != rows[TP]['hardware']['host'], 'Require two hosts with 16 distinct GPUs')
    require(len({r['global_tokens_sha256'] for r in rows}) == 1 and
            rows[0]['tokens_sha256'] != rows[TP]['tokens_sha256'], 'DP replicas must consume distinct global-batch partitions')
    for node in range(DP):
        group = rows[node*TP:(node+1)*TP]
        require(len({r['tokens_sha256'] for r in group}) == 1 and all(r['losses'] == group[0]['losses'] for r in group),
                'TP replicas disagree on inputs/loss')
    for tp in range(TP):
        a,b = rows[tp],rows[tp+TP]
        require(a['initial_parameters_sha256'] == b['initial_parameters_sha256'] and
                a['final_parameters_sha256'] == b['final_parameters_sha256'], 'DP parameter replicas diverged')
        if job['mode'] == 'smoke':
            require(len(a['parameter_trajectory']) == len(b['parameter_trajectory']) == 3 and
                    a['parameter_trajectory'] == b['parameter_trajectory'] and
                    all(len(x['gradients']) == len(x['parameters']) == 389 for x in a['parameter_trajectory']),
                    'DP gradients/updates are not synchronized for every parameter')
    return rows


def paired(candidate, reference):
    require(len(candidate) == len(reference) >= 6, 'Incomplete paired blocks')
    logs = [math.log(a/b) for a,b in zip(candidate, reference)]
    rng = random.Random(20261028)
    samples = sorted(100*(1-math.exp(statistics.mean(rng.choices(logs, k=len(logs))))) for _ in range(10000))
    return dict(percent=100*(1-math.exp(statistics.mean(logs))), ci95=[samples[249],samples[9749]], paired_blocks=len(logs))


def report(samples, trials, blocks, package_id):
    expected = schedule(trials,blocks)
    require(len(samples) == len(expected) and all(all(s[k] == j[k] for k in j) for s,j in zip(samples,expected)),
            'Missing/reordered formal windows')
    comparisons = {}
    for trial in range(1,trials+1):
        values = {arm:[s['ms_per_step'] for s in samples if s['trial']==trial and s['arm']==arm] for arm in ARMS}
        require(all(len(v)==blocks and all(math.isfinite(x) and x>0 for x in v) for v in values.values()), 'Invalid timing windows')
        comparisons[f'compare{trial}'] = dict(mean_ms={a:statistics.mean(v) for a,v in values.items()},
            median_ms={a:statistics.median(v) for a,v in values.items()},
            tokens_per_second={a:8192000/statistics.mean(v) for a,v in values.items()},
            comparisons={f'ma11/{a}':paired(values['ma11'],values[a]) for a in ARMS[1:]})
    return dict(protocol=PROTOCOL, package_id=package_id, completed=True, correctness_passed=True,
                execution_path_validated=True, model=MODEL, trials=trials, blocks=blocks,
                all_samples=samples, comparisons=comparisons, formal_optimizer_steps=len(samples)*50,
                timing_rank_records=len(samples)*WORLD, all_windows_retained=True,
                limitation='Synthetic repeated global batch; different quantization policies need not have identical trajectories. '
                           'Each trial is assessed separately; no historical timing samples.')
