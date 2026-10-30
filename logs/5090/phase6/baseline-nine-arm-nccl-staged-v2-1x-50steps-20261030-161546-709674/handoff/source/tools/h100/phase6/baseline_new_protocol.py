"""Five-arm single-node full-step contracts and paired statistics; CPU only."""
import math
import random
from statistics import mean, median

from tp8_dp2_protocol import digest, paired, read, require, sha, write, validate_projection_checks

PROTOCOL = 'h100-baseline-new-five-arm-v1'
ARMS = ('ma11', 'ma89', 'cr3', 'flux-basic', 'megatron-cuda')
SELECTIVE = ARMS[:3]
LABELS = dict(zip(ARMS, ('MA11', 'MA89', 'CR3', 'Flux＋基础量化', 'Megatron＋CUDA量化')))
POLICY = dict.fromkeys(SELECTIVE, 'interleaved_arrival_selective_attn_mlp') | {
    'flux-basic': 'basic', 'megatron-cuda': 'native_cuda_taco_attn_mlp'}
NATIVE_ARM = {'ma11': 'F4-cache-zcount-csoftmax', 'ma89': 'F4-cache-zcount', 'cr3': 'F4'}
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
             micro_batch=1, global_batch=4, tp=8, dp=1, sm_count=132,
             seed=1234, token_seed=4321, vocab=32768)
TIMING = dict(warmup_steps=10, timed_steps=50, smoke_warmup_steps=2, smoke_steps=3)
SEED = 20261029


def schedule(trials=1, blocks=10):
    require(type(trials) is int and trials in (1, 2) and type(blocks) is int and
            blocks >= 10 and blocks % 10 == 0, 'Use 1/2 trials and a multiple of 10 blocks')
    # Odd-sized Williams design: five rotations and their reversals. Every arm
    # occupies each slot twice; each pair has five observations in either order.
    base = (0, 1, 4, 2, 3)
    orders = [tuple(ARMS[(i+k) % 5] for i in base) for k in range(5)]
    orders += [tuple(reversed(order)) for order in orders[:]]
    result = []
    for trial in range(1, trials+1):
        rng = random.Random(SEED+trial)
        for cycle in range(blocks//10):
            shuffled = list(orders); rng.shuffle(shuffled)
            for j, order in enumerate(shuffled):
                block = cycle*10+j
                for slot, arm in enumerate(order):
                    result.append(dict(trial=trial, block=block, index=block*5+slot,
                        arm=arm, mode='timing', directory=f'compare{trial}/block{block:02d}-{arm}'))
    return result


def jobs(trials=1, blocks=10):
    special = lambda mode: [dict(trial=0, block=-1, index=i, arm=arm, mode=mode,
                                  directory=f'{mode}-{arm}') for i, arm in enumerate(ARMS)]
    return special('smoke')+schedule(trials, blocks)+special('profile')


def identity(rows):
    keys = ('rank', 'gpu_uuid', 'torch', 'cuda', 'nccl', 'python', 'hardware', 'model',
            'initial_parameters_sha256', 'initial_rng_sha256', 'tokens_sha256',
            'parameter_count_local', 'padded_vocab_size', 'communication')
    return [{key: row[key] for key in keys} for row in rows]


def equivalent(rows, other, smoke=False):
    keys = ['losses', 'grad_norms', 'final_parameters_sha256']
    if smoke: keys += ['parameter_trajectory', 'trainable_parameters']
    require([{k:r[k] for k in keys} for r in rows] == [{k:r[k] for k in keys} for r in other],
            'Selective arms changed loss, gradient or parameter-update identity')


def validate_rows(rows, job, plan, worker):
    from ma4_f_fullstep import validate_parameter_trajectory
    from ma4_f_bench import validate_consumer
    from ma11_fullstep import validate_activation_setup, validate_zero_count_state, validate_softmax_state
    import baseline_codec_validation
    import basic_quant_validation
    require(len(rows) == 8 and {r['rank'] for r in rows} == set(range(8)), 'Missing/duplicate rank')
    rows = sorted(rows, key=lambda r:r['rank'])
    require(len({r['gpu_uuid'] for r in rows}) == 8, 'Eight distinct GPUs required')
    arm, mode = job['arm'], job['mode']
    warmup, count = (2, 3) if mode == 'smoke' else (10, 1 if mode == 'profile' else 50)
    expected_modules = {f'decoder.layers.{i}.{suffix}' for i in range(32)
                        for suffix in ('self_attention.linear_proj', 'mlp.linear_fc2')}
    cfg = worker['config']
    for rank, row in enumerate(rows):
        require(row['protocol'] == PROTOCOL and row['run_id'] == plan['run_id'] and
                row['arm'] == arm and row['mode'] == mode and row['block'] == job['block'] and
                row['window'] == job['index'] and row['trial'] == job['trial'] and
                row['worker_config_sha256'] == worker['files']['config.json'] and
                row['completed'] is True and row['passed'] is True and row['skips'] == 0,
                'Worker identity/completion changed')
        require(row['world_size'] == 8 and row['local_rank'] == row['tp_rank'] == rank and
                row['dp_rank'] == 0 and row['tp_group_ranks'] == list(range(8)) and
                row['dp_group_ranks'] == [rank], 'Expected single-node TP8 DP1')
        require(all(row['model'][k] == v for k,v in MODEL.items() if k != 'sm_count') and
                row['precision'] == 'bf16' and row['torch'] == '2.6.0+cu124' and
                row['cuda'] == '12.4' and row['nccl'] == [2,21,5] and
                row['policy'] == POLICY[arm] and row['communication'] == plan['communication'] and
                row['scope_definition'] == worker['definition'] and
                row['role_plan_sha256'] == cfg['role_plan_sha256'], 'Model/software/route changed')
        require(row['sampling'] is False and row['profiling'] == (mode == 'profile') and
                row['tokens_per_optimizer_step'] == 8192 and row['warmup_steps'] == warmup and
                row['timing_protocol'] == 'full-step-one-clear-v2' and
                len(row['losses']) == len(row['grad_norms']) == count and
                all(math.isfinite(x) for x in row['losses']+row['grad_norms']) and
                row['losses'] == rows[0]['losses'], 'Workload/timing or finite TP loss checks failed')
        require(set(row['target_modules']) == expected_modules == set(row['timed_calls']) ==
                set(row['target_calls']) == set(row['enabled_modules']) and
                row['independent_scope_instances'] == row['replaced_modules'] == 64 and
                all(v == count*4 for v in row['timed_calls'].values()) and
                row['warmup_calls'] == row['target_calls'] == dict.fromkeys(expected_modules,warmup*4) and
                row['fallback_calls'] == 0, 'Missing projection coverage or fallback')
        route = 'candidate' if arm in SELECTIVE else 'flux_tensor' if arm=='flux-basic' else 'tensor'
        require(row['module_policies'] == row['post_audit_routes'] == dict.fromkeys(expected_modules,route),
                'Measured module route changed')
        validate_projection_checks(row, mode)
        if mode == 'smoke':
            for check in row['forward_checks'].values():
                require(check['budget_vs_bf16'] == .05 and check['budget_vs_codec'] == .01 and
                        all(math.isfinite(check[k]) for k in ('max_abs','relative_l2','protocol_relative_l2')) and
                        0 <= check['relative_l2'] <= .05 and 0 <= check['protocol_relative_l2'] <= .01,
                        'Projection numerical budget changed')
            require(len(row['trainable_parameters']) == 389, 'Incomplete parameter universe')
            require({n+s for n in expected_modules for s in ('.weight','.bias')} <= set(row['trainable_parameters']),
                    'Missing target weight/bias gradients')
            validate_parameter_trajectory(row['parameter_trajectory'], row['trainable_parameters'])
        else:
            require(row['parameter_trajectory'] == [], 'Parameter audit leaked into timing/profile')
        h = row['final_parameters_sha256']
        require(isinstance(h,str) and len(h)==64 and all(c in '0123456789abcdef' for c in h),
                'Missing final parameter fingerprint')
        for key in ('peak_allocated_gib','peak_reserved_gib'):
            require(math.isfinite(row[key]) and row[key]>0, 'Invalid memory peak')
        if mode != 'profile':
            elapsed = row['elapsed_seconds']
            require(row['timed_steps'] == count and math.isfinite(elapsed) and elapsed>0 and
                    math.isclose(row['ms_per_step'],elapsed*1000/count,rel_tol=1e-10), 'Invalid wall-clock sample')
        for name, expected in worker['libraries'].items():
            require(row[name] == expected, 'Wrong loaded native binary: '+name)
        if arm == 'megatron-cuda':
            require(row['flux_library_loaded'] is False and row['native_linear_forward_preserved'] is True,
                    'Megatron baseline replaced its original GEMM')
        if arm in SELECTIVE:
            prefix = 'ma4_f' if arm == 'cr3' else 'ma11'
            consumer = row[prefix+'_consumer_state']
            require(set(consumer) == expected_modules, 'Missing CR3 consumer modules')
            for name, state in consumer.items():
                require(state['projection_calls'] == state['bda_calls'] == (warmup+count)*4 and
                        state['bias_parameter_id']>0 and
                        sum(x['successful_calls'] for x in state['native'])==(warmup+count)*4,
                        'Consumer calls missing')
                validate_consumer(state['native'],row[prefix+'_submission_state'][name],'F4')
            if arm != 'cr3':
                validate_activation_setup(row['ma11_activation_setup'], expected_modules)
                validate_zero_count_state(row['ma11_zero_count_state'], NATIVE_ARM[arm], warmup+count, mode, rank)
                validate_softmax_state(row['ma11_softmax_state'], NATIVE_ARM[arm], warmup+count, mode)
            else:
                require(not any(k in row for k in ('ma11_activation_setup','ma11_zero_count_state','ma11_softmax_state')),
                        'CR3 has extra MA8/MA9/MA11 wrappers')
        else:
            require(row['codec_identity'] == cfg['codec_identity'] and row['codec_library'] == cfg['codec_library'] and
                    set(row['codec_calls']) == expected_modules and
                    all(v == (warmup+count)*4 for v in row['codec_calls'].values()), 'Wrong standalone codec')
            if mode == 'smoke':
                if arm == 'flux-basic':
                    basic_quant_validation.verify(row['codec_validation'],cfg['codec_library'],rank)
                    require(set(row['retained_output_checks']) == {'attn','mlp'} and
                            all(x['passed'] and x['independent_storage'] and x['unchanged_after_subsequent_call']
                                for x in row['retained_output_checks'].values()), 'Basic output ownership failed')
                else: baseline_codec_validation.verify(row['codec_validation'],cfg['codec_library'])
            else: require(row['codec_validation'] is None, 'Codec audit leaked into timing/profile')
    return rows


def summarize(rows, job):
    return dict(job, ms_per_step=max(r['elapsed_seconds'] for r in rows)*1000/50,
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows))


def report(samples, trials=1, blocks=10):
    expected = schedule(trials, blocks)
    require(len(samples)==len(expected) and all(all(s[k]==v for k,v in j.items())
            for s,j in zip(samples,expected)), 'Missing/reordered five-arm windows')
    comparisons = {}
    for trial in range(1,trials+1):
        group = [s for s in samples if s['trial']==trial]
        values = {arm:[s['ms_per_step'] for s in group if s['arm']==arm] for arm in ARMS}
        require(all(len(v)==blocks and all(math.isfinite(x) and x>0 for x in v) for v in values.values()),
                'Invalid baseline timing samples')
        means = {a:mean(v) for a,v in values.items()}
        stats = {a:dict(mean_ms=means[a],median_ms=median(values[a]),min_ms=min(values[a]),
                       max_ms=max(values[a]),tokens_per_second=8192000/means[a],
                       ma11_mean_reduction_percent=100*(1-means['ma11']/means[a]),
                       peak_allocated_gib=mean(s['peak_allocated_gib'] for s in group if s['arm']==a),
                       peak_reserved_gib=mean(s['peak_reserved_gib'] for s in group if s['arm']==a)) for a in ARMS}
        pairs = {a+'/'+b:paired(values[a],values[b]) for i,a in enumerate(ARMS) for b in ARMS[i+1:]}
        comparisons[f'compare{trial}'] = dict(arms=stats,pairs=pairs)
    return dict(protocol=PROTOCOL,completed=True,trials=trials,blocks=blocks,arms=list(ARMS),model=MODEL,
                timing_windows=len(samples),timing_rank_records=len(samples)*8,formal_optimizer_steps=len(samples)*50,
                comparisons=comparisons,all_samples=samples,all_windows_retained=True,
                limitation='Single-node TP8 DP1, fixed synthetic input. Different quantization policies are not bitwise-equivalent. '
                           'Trials reported separately; positive or negative performance never stops the campaign.')


def markdown(result):
    lines = ['# 新服务器五组 baseline', '', '完整 optimizer step；均值使用全部正式窗口，正降幅表示 MA11 更快。', '']
    for trial, data in result['comparisons'].items():
        lines += [f'## {trial}', '', '| 组别 | 平均 ms/step | 中位数 | 最慢窗口 | tokens/s | MA11 相对平均耗时下降 | allocated 峰值 GiB |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm,x in data['arms'].items():
            lines.append(f"| {LABELS[arm]} | {x['mean_ms']:.3f} | {x['median_ms']:.3f} | {x['max_ms']:.3f} | "
                         f"{x['tokens_per_second']:.2f} | {x['ma11_mean_reduction_percent']:+.3f}% | {x['peak_allocated_gib']:.6f} |")
        lines += ['', '| 同期配对（前者相对后者） | 配对降幅 %［95% CI］ |', '| --- | --- |']
        for pair,x in data['pairs'].items():
            a,b=pair.split('/')
            lines.append(f"| {LABELS[a]} / {LABELS[b]} | {x['percent']:+.3f}［{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}］ |")
        lines.append('')
    return '\n'.join(lines+[result['limitation'],''])
