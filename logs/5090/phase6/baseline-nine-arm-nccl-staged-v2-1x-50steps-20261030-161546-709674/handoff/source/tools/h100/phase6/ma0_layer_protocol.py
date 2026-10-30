"""CPU-only contracts and paired statistics for complete TP8 sublayers."""
import math
from statistics import mean

from tp8_dp2_protocol import paired, require, read, write, sha, digest

PROTOCOL = 'h100-ma0-complete-sublayer-v1'
ARMS = ('flux-basic', 'ma11')
ROLES = ('attention', 'mlp')
PHASES = ('forward_ms', 'backward_ms', 'total_ms')
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
             micro_batch=1, global_batch=4, tp=8, dp=1, sm_count=132,
             seed=1234, token_seed=4321, vocab=32768)
LIMITATION = ('Isolated complete sublayers at fixed input/weights, including autograd TP communication; '
              'not in-training overlap, optimizer time, or an additive decomposition of full-step savings. '
              'Step-level cross-module gradient finalization is outside these regions. '
              'MA11 and basic quantization have different quantization policies, so cross-arm bitwise '
              'equality is not required. MA9 optimizer statistics are outside these regions.')


def schedule(blocks=8):
    require(type(blocks) is int and blocks >= 8 and blocks % 2 == 0, 'Use an even number of blocks >=8')
    return [dict(arm=arm, mode='timing', block=b, index=2*b+i,
                 directory=f'block{b:02d}-{arm}')
            for b in range(blocks) for i, arm in enumerate(ARMS if b % 2 == 0 else ARMS[::-1])]


def validate_rows(rows, job, plan):
    require(len(rows) == 8 and {r['rank'] for r in rows} == set(range(8)), 'Missing/duplicate TP ranks')
    rows = sorted(rows, key=lambda r:r['rank'])
    require(len({r['gpu_uuid'] for r in rows}) == 8, 'Eight distinct H100 GPUs required')
    for row in rows:
        require(row['protocol'] == PROTOCOL and row['arm'] == job['arm'] and
                row['run_id'] == plan['run_id'] and row['block'] == job['block'] and
                row['window'] == job['index'] and row['mode'] == job['mode'] and
                row['completed'] is True and row['passed'] is True and
                row['world_size'] == 8 and row['local_rank'] == row['tp_rank'] == row['rank'] and
                row['dp_rank'] == 0 and row['tp_group_ranks'] == list(range(8)) and
                row['dp_group_ranks'] == [row['rank']], 'Worker identity/topology changed')
        if job['mode'] == 'smoke':
            require(row['warmup_steps'] == 2 and row['timed_steps'] == 3 and row['skips'] == 0 and
                    row['fallback_calls'] == 0 and len(row['target_modules']) == 64 and
                    set(row['forward_checks']) == set(row['target_modules']) and
                    all(v.get('passed') is True for v in row['forward_checks'].values()) and
                    len(row['losses']) == len(row['grad_norms']) == 3 and
                    all(math.isfinite(v) for v in row['losses']+row['grad_norms']),
                    'Original full-model smoke/numeric checks incomplete')
            if job['arm'] == 'ma11':
                require(row['ma11_activation_setup']['installed'] and
                        row['ma11_softmax_state']['self_test']['passed'] and
                        row['ma11_zero_count_state']['self_test']['passed'] and
                        len(row['ma11_consumer_state']) == 64, 'MA11 mechanism preflight missing')
            else:
                require(row['codec_validation']['passed'] and not row['codec_validation']['synthetic'] and
                        set(row['retained_output_checks']) == {'attn','mlp'} and
                        all(v['passed'] for v in row['retained_output_checks'].values()),
                        'Basic receiver/independent output preflight missing')
            continue
        require(row['weights_unchanged'] is True and row['instrumentation'] == 'cuda_events_no_profiler' and
                row['warmup_iterations'] == plan['warmup'] and row['iterations'] == plan['iterations'] and
                not row['forward_checks'] and row['mechanism_validated'] is True,
                'Timing boundary, weights or mechanism changed')
        regions = row['regions']
        require(set(regions) == {f'{role}/{layer}' for role in ROLES for layer in range(32)},
                'Complete 32-layer Attention/MLP coverage required')
        for region in regions.values():
            require(region['boundary_equivalent'] and region['finite_gradients'] and
                    region['gradient_parameters'] > 0 and len(region['samples']) == plan['iterations'],
                    'Incomplete region boundary, gradients or samples')
            for sample in region['samples']:
                require(set(sample) == set(PHASES) and all(math.isfinite(v) and v > 0 for v in sample.values()) and
                        abs(sample['total_ms']-sample['forward_ms']-sample['backward_ms']) < 1e-3,
                        'Invalid CUDA event sample')
    return rows


def identity(rows):
    return [{k:r[k] for k in ('rank','gpu_uuid','torch','cuda','nccl','python',
                              'initial_parameters_sha256','initial_rng_sha256','tokens_sha256')}
            for r in rows]


def region_identity(rows):
    return [{key:{field:region[field] for field in ('input_sha256','grad_output_sha256','weights_sha256')}
             for key,region in r['regions'].items()} for r in rows]


def summarize_window(rows):
    # Rank maxima are taken at each matching iteration. The total uses the
    # SAME rank's fwd+bwd interval, not max(fwd)+max(bwd) from different ranks.
    return {key:{phase:mean(max(r['regions'][key]['samples'][i][phase] for r in rows)
                            for i in range(len(rows[0]['regions'][key]['samples'])))
                 for phase in PHASES} for key in rows[0]['regions']}


def comparison(reference, candidate):
    return dict(reference_ms=mean(reference), candidate_ms=mean(candidate),
                saved_ms=mean(reference)-mean(candidate), **paired(candidate, reference))


def report(samples, blocks):
    require(len(samples) == blocks*2 and
            {(x['block'],x['arm']) for x in samples} == {(b,a) for b in range(blocks) for a in ARMS},
            'Incomplete balanced blocks')
    by = {(x['block'],x['arm']):x['regions'] for x in samples}
    layers = {f'{role}/{layer}':{phase:comparison(
        [by[b,ARMS[0]][f'{role}/{layer}'][phase] for b in range(blocks)],
        [by[b,ARMS[1]][f'{role}/{layer}'][phase] for b in range(blocks)]) for phase in PHASES}
        for role in ROLES for layer in range(32)}
    roles = {role:{phase:comparison(
        [mean(by[b,ARMS[0]][f'{role}/{layer}'][phase] for layer in range(32)) for b in range(blocks)],
        [mean(by[b,ARMS[1]][f'{role}/{layer}'][phase] for layer in range(32)) for b in range(blocks)])
        for phase in PHASES} for role in ROLES}
    return dict(protocol=PROTOCOL, completed=True, blocks=blocks, arms=list(ARMS),
                model=MODEL, roles=roles, layers=layers, all_samples=samples, limitation=LIMITATION)


def markdown(result):
    lines = ['# MA0：MA11 与 Flux＋基础量化完整子层', '',
             '单位 ms/次；每次调用是一个 microbatch 的一个完整子层。正降幅表示 MA11 更快。', '',
             '| 范围 | 阶段 | Flux＋基础量化 | MA11 | 节省 ms | 配对降幅 % [95% CI] |',
             '| --- | --- | ---: | ---: | ---: | --- |']
    for name, regions in [('32层平均/'+k,v) for k,v in result['roles'].items()] + list(result['layers'].items()):
        for phase,x in regions.items():
            lines.append(f"| {name} | {phase.removesuffix('_ms')} | {x['reference_ms']:.6f} | "
                         f"{x['candidate_ms']:.6f} | {x['saved_ms']:.6f} | "
                         f"{x['percent']:+.3f} [{x['ci95'][0]:+.3f}, {x['ci95'][1]:+.3f}] |")
    return '\n'.join(lines+['',LIMITATION,''])
