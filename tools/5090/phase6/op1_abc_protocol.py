"""5090 staged interleave / arrival / selective full-step comparison."""
import itertools
import math
import random
from statistics import mean, median
from rtx5090_loader import load
PROTOCOL = 'rtx5090-phase6-op1-abc-staged-v1'
ARMS = ('flux-basic-interleaved', 'flux-basic-arrival', 'flux-basic-selective', 'flux-improved')


def definition(arm):
    from op1_abc_routing import definition as describe
    return describe(arm)

base = load('protocol')
NATIVE = {'flux-improved': 'c'}
BASIC_ARMS = ARMS[:3]
LABELS = dict(zip(ARMS, ('Flux＋基础量化＋交替排序（A）',
    'Flux＋基础量化＋交替排序＋到达优先（B）',
    'Flux＋基础量化＋交替排序＋到达优先＋选择性量化（C）', 'Flux＋改进量化')))
POLICY = {arm: arm for arm in ARMS}
MODEL, TIMING, TOKENS_PER_STEP = base.MODEL, base.TIMING, base.TOKENS_PER_STEP
TRANSPORT, TRANSPORT_ENV = base.TRANSPORT, base.TRANSPORT_ENV
BLOCKS_PER_CYCLE = 8
DEFAULT_BLOCKS = 3
EQUIVALENCE_GROUPS = ()
read, write, require, sha, digest, paired = base.read, base.write, base.require, base.sha, base.digest, base.paired


def schedule(trials=1, blocks=DEFAULT_BLOCKS):
    require(type(trials) is int and trials in (1, 2) and type(blocks) is int
            and blocks > 0 and (blocks == 3 or blocks % 8 == 0),
            'Use 1/2 trials and either 3 blocks or a positive multiple of 8 blocks')
    seed_order = (0, 1, 3, 2)
    cycle_orders = [tuple(ARMS[(i+k) % 4] for i in seed_order) for k in range(4)]
    cycle_orders += [order[::-1] for order in cycle_orders[:]]
    result = []
    for trial in range(1, trials+1):
        rng = random.Random(20261101+trial)
        for cycle in range((blocks+7)//8):
            orders = list(cycle_orders)
            rng.shuffle(orders)
            for offset, order in enumerate(orders):
                block = cycle*8+offset
                if block >= blocks:
                    break
                for slot, arm in enumerate(order):
                    result.append(dict(trial=trial, block=block, index=block*4+slot, arm=arm,
                                       mode='timing', directory=f'compare{trial}/block{block:02d}-{arm}'))
    return result


def jobs(trials=1, blocks=DEFAULT_BLOCKS):
    return [dict(trial=0, block=-1, index=i, arm=arm, mode=mode, directory=f'{mode}-{arm}')
            for mode in ('smoke', 'profile') for i, arm in enumerate(ARMS)] + schedule(trials, blocks)


def summarize(rows, job):
    require(job['arm'] in ARMS, 'Unknown four-arm sample')
    return dict(base.summarize(rows, dict(job, arm='flux-basic')), **job)


def paired_summary(candidate, reference):
    if len(candidate) >= 6:
        return paired(candidate, reference)
    require(len(candidate) == len(reference) == 3
            and all(math.isfinite(x) and x > 0 for x in (*candidate, *reference)),
            'Three complete positive paired measurements required for the short run')
    return dict(percent=100*(1-math.exp(mean(math.log(a/b) for a, b in zip(candidate, reference)))),
                ci95=None, paired_blocks=3)


def report(samples, trials=1, blocks=DEFAULT_BLOCKS):
    expected = schedule(trials, blocks)
    require(len(samples) == len(expected) and all(all(s.get(k) == v for k, v in job.items())
            for s, job in zip(samples, expected)), 'Missing/reordered/foreign four-arm windows')
    for row in samples:
        require(all(math.isfinite(row[k]) and row[k] > 0 for k in
                    ('ms_per_step', 'peak_allocated_gib', 'peak_reserved_gib')), 'Invalid sample')
        require(math.isfinite(row['native_workspace_gib']) and row['native_workspace_gib'] >= 0
                and math.isfinite(row['peak_allocated_plus_taco_workspace_gib'])
                and row['peak_allocated_plus_taco_workspace_gib'] >= row['peak_allocated_gib'], 'Invalid memory')
    comparisons = {}
    for trial in range(1, trials+1):
        own = [row for row in samples if row['trial'] == trial]
        values = {arm: [row['ms_per_step'] for row in own if row['arm'] == arm] for arm in ARMS}
        means = {arm: mean(values[arm]) for arm in ARMS}
        stats = {arm: dict(mean_ms=means[arm], median_ms=median(values[arm]), min_ms=min(values[arm]),
            max_ms=max(values[arm]), tokens_per_second=TOKENS_PER_STEP*1000/means[arm],
            reduction_vs_improved_percent=100*(1-means[arm]/means['flux-improved']),
            **{'max_'+key: max(row[key] for row in own if row['arm'] == arm) for key in
               ('peak_allocated_gib', 'peak_reserved_gib', 'native_workspace_gib',
                'peak_allocated_plus_taco_workspace_gib')}) for arm in ARMS}
        pairs = {a+'/'+b: dict(paired_summary(values[a], values[b]),
                    faster_blocks=sum(x < y for x, y in zip(values[a], values[b])))
                 for a, b in itertools.combinations(ARMS, 2)}
        comparisons[f'compare{trial}'] = dict(arms=stats, pairs=pairs)
    return dict(protocol=PROTOCOL, transport=TRANSPORT, completed=True, arms=list(ARMS),
        definitions={arm: definition(arm) for arm in ARMS}, trials=trials, blocks=blocks,
        counterbalanced_order=blocks % BLOCKS_PER_CYCLE == 0, pairwise_ci_available=blocks >= 6,
        model=MODEL, timing=TIMING, timing_windows=len(samples), timing_rank_records=len(samples)*8,
        formal_optimizer_steps=len(samples)*50, all_windows_retained=True, comparisons=comparisons,
        all_samples=samples, numerical_equivalence_required=False, reduction_changes_allowed=True,
        original_peer_overlap_preserved=False,
        limitation='Fixed single-node RTX5090 TP8 workload. A exchanges one complete packet; B/C exchange '
                   'four output-row waves and enqueue each decoder after its NCCL completion event. '
                   'GEMM completes before communication; no original cross-GPU tile-arrival overlap. '
                   'C quantizes a deterministic last 1/64 of groups and keeps the remainder BF16, without measured 5090 arrival calibration. '
                   'Flux improved retains the separately documented original-order fused implementation. '
                   'GEMM, codec, decoder, transport and rounding can differ; gains are end-to-end combinations. '
                   'No historical pooling or filtering of negative gains. '
                   + ('The three-block short run does not complete the balanced ordering cycle; '
                      'paired point estimates only, without confidence intervals.' if blocks == 3 else
                      'Six paired CIs are unadjusted.'))



def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete/foreign report')
    lines = ['# RTX 5090 OP1 A/B/C 与 Flux 改进量化完整训练对比', '', f"协议 `{PROTOCOL}`；{result['trials']} trial × "
        f"{result['blocks']} blocks × 4 组，共 {result['timing_windows']} 窗口、"
        f"{result['formal_optimizer_steps']} 个正式 optimizer step。", '']
    if result['blocks'] == 3:
        lines += ['三块短测已完成计划窗口，但未完成八块排序平衡周期；仅报告描述统计与配对点估计，不计算 95% CI。', '']
    for trial, record in result['comparisons'].items():
        lines += [f'## {trial}', '',
            '| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对改进量化降幅 | 最大 allocated GiB | allocated＋native GiB |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            v = record['arms'][arm]
            lines.append(f"| {LABELS[arm]} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | "
                f"{v['max_ms']:.3f} | {v['tokens_per_second']:.2f} | {v['reduction_vs_improved_percent']:+.3f}% | "
                f"{v['max_peak_allocated_gib']:.6f} | {v['max_peak_allocated_plus_taco_workspace_gib']:.6f} |")
        lines += ['', '| 配对（前者 / 后者） | 降幅 % [95% CI] | 更快块数 |', '| --- | ---: | ---: |']
        for key, v in record['pairs'].items():
            a, b = key.split('/')
            interval = (f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" if v['ci95'] is not None
                        else '［三块短测，不计算 CI］')
            lines.append(f"| {LABELS[a]} / {LABELS[b]} | {v['percent']:+.3f} "
                f"{interval} | {v['faster_blocks']}/{v['paired_blocks']} |")
    lines += ['', '全部窗口算术均值；配对估计使用完整块对数比。保留负收益。'
              + ('三块短测不计算置信区间。' if result['blocks'] == 3 else '区间使用 10,000 次 bootstrap。'),
              'allocated＋native 不含 CUDA context、NCCL 等内部占用；逻辑贡献字节数不是 NCCL 实测链路流量。',
              '各组独立通过数值和完整训练检查；不要求不同量化或归约的轨迹逐位一致。']
    for key, value in result.get('numerical_comparisons', {}).items():
        lines.append(f"数值观测 `{key}`：loss 最大绝对差 {value['max_loss_abs_difference']:.6g}；"
                     f"gradient norm 最大绝对差 {value['max_grad_norm_abs_difference']:.6g}。")
    return '\n'.join(lines+['', result['limitation'], ''])
