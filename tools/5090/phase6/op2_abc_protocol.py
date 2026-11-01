"""Three ABC paths with shared MA89 and MA11 optimizations."""
import itertools
import math
import random
from statistics import mean, median
from rtx5090_loader import load
PROTOCOL = 'rtx5090-phase6-op2-abc-ma11-v1'
ARMS = ('flux-basic-interleaved', 'flux-basic-arrival', 'flux-basic-selective')


def definition(arm):
    from op2_abc_sources import definition as describe
    return describe(arm)

base = load('protocol')
NATIVE = {}
BASIC_ARMS = ARMS[:3]
LABELS = dict(zip(ARMS, ('A＋MA89＋MA11', 'B＋MA89＋MA11', 'C＋MA89＋MA11')))
POLICY = {arm: arm for arm in ARMS}
MODEL, TIMING, TOKENS_PER_STEP = base.MODEL, base.TIMING, base.TOKENS_PER_STEP
TRANSPORT, TRANSPORT_ENV = base.TRANSPORT, base.TRANSPORT_ENV
BLOCKS_PER_CYCLE = 6
DEFAULT_BLOCKS = 2
EQUIVALENCE_GROUPS = ()
read, write, require, sha, digest, paired = base.read, base.write, base.require, base.sha, base.digest, base.paired


def schedule(trials=1, blocks=DEFAULT_BLOCKS):
    require(type(trials) is int and trials in (1, 2) and type(blocks) is int
            and blocks > 0 and (blocks in (2, 3) or blocks % 6 == 0),
            'Use 1/2 trials and either 2/3 blocks or a positive multiple of 6 blocks')
    cycle_orders = list(itertools.permutations(ARMS))
    result = []
    for trial in range(1, trials+1):
        rng = random.Random(20261101+trial)
        for cycle in range((blocks+5)//6):
            orders = list(cycle_orders)
            rng.shuffle(orders)
            for offset, order in enumerate(orders):
                block = cycle*6+offset
                if block >= blocks:
                    break
                for slot, arm in enumerate(order):
                    result.append(dict(trial=trial, block=block, index=block*3+slot, arm=arm,
                                       mode='timing', directory=f'compare{trial}/block{block:02d}-{arm}'))
    return result


def jobs(trials=1, blocks=DEFAULT_BLOCKS):
    return [dict(trial=0, block=-1, index=i, arm=arm, mode=mode, directory=f'{mode}-{arm}')
            for mode in ('smoke', 'profile') for i, arm in enumerate(ARMS)] + schedule(trials, blocks)


def summarize(rows, job):
    require(job['arm'] in ARMS, 'Unknown three-arm sample')
    return dict(base.summarize(rows, dict(job, arm='flux-basic')), **job)


def paired_summary(candidate, reference):
    if len(candidate) >= 6:
        return paired(candidate, reference)
    require(len(candidate) == len(reference) and len(candidate) in (2, 3)
            and all(math.isfinite(x) and x > 0 for x in (*candidate, *reference)),
            'Two or three complete positive paired measurements required for the short run')
    return dict(percent=100*(1-math.exp(mean(math.log(a/b) for a, b in zip(candidate, reference)))),
                ci95=None, paired_blocks=len(candidate))


def report(samples, trials=1, blocks=DEFAULT_BLOCKS):
    expected = schedule(trials, blocks)
    require(len(samples) == len(expected) and all(all(s.get(k) == v for k, v in job.items())
            for s, job in zip(samples, expected)), 'Missing/reordered/foreign three-arm windows')
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
            reduction_vs_a_percent=100*(1-means[arm]/means[ARMS[0]]),
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
        mechanism_stack=['ma8_activation_cache', 'ma9_zero_count', 'ma11_softmax'],
        increment_over_unoptimized_validated=False,
        limitation='Three ABC projection paths with the same MA8 cache, MA9 zero statistic and MA11 softmax. '
                   'MA11 already includes MA89: shared improvements are installed once. '
                   'No CR3 consumer fusion is transplanted. Original ABC codec, chunks and CUDA connections stay fixed. '
                   'No contemporaneous unoptimized or MA89-only control: this run cannot isolate their incremental gains. '
                   'No historical pooling; all windows and negative gains retained. '
                   + (f'{blocks}-block short run has incomplete counterbalancing and no confidence interval.'
                      if blocks in (2, 3) else 'Three paired confidence intervals are unadjusted.'))




def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete/foreign report')
    lines = ['# RTX 5090 OP2：A/B/C 叠加 MA89 与 MA11', '', f"协议 `{PROTOCOL}`；{result['trials']} trial × "
        f"{result['blocks']} blocks × 3 组，共 {result['timing_windows']} 窗口、"
        f"{result['formal_optimizer_steps']} 个正式 optimizer step。", '']
    if result['blocks'] in (2, 3):
        lines += [f"{result['blocks']} 块短测已完成计划窗口，但未完成六块排序平衡周期；仅报告描述统计与配对点估计，不计算 95% CI。", '']
    for trial, record in result['comparisons'].items():
        lines += [f'## {trial}', '',
            '| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对 A＋MA89＋MA11 降幅 | 最大 allocated GiB | allocated＋native GiB |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            v = record['arms'][arm]
            lines.append(f"| {LABELS[arm]} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | "
                f"{v['max_ms']:.3f} | {v['tokens_per_second']:.2f} | {v['reduction_vs_a_percent']:+.3f}% | "
                f"{v['max_peak_allocated_gib']:.6f} | {v['max_peak_allocated_plus_taco_workspace_gib']:.6f} |")
        lines += ['', '| 配对（前者 / 后者） | 降幅 % [95% CI] | 更快块数 |', '| --- | ---: | ---: |']
        for key, v in record['pairs'].items():
            a, b = key.split('/')
            interval = (f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" if v['ci95'] is not None
                        else f"［{result['blocks']} 块短测，不计算 CI］")
            lines.append(f"| {LABELS[a]} / {LABELS[b]} | {v['percent']:+.3f} "
                f"{interval} | {v['faster_blocks']}/{v['paired_blocks']} |")
    lines += ['', '全部窗口算术均值；配对估计使用完整块对数比。保留负收益。'
              + (f"{result['blocks']} 块短测不计算置信区间。" if result['blocks'] in (2, 3) else '区间使用 10,000 次 bootstrap。'),
              'allocated＋native 不含 CUDA context、NCCL 等内部占用；逻辑贡献字节数不是 NCCL 实测链路流量。',
              '各组独立通过数值和完整训练检查；不要求不同量化或归约的轨迹逐位一致。']
    for key, value in result.get('numerical_comparisons', {}).items():
        lines.append(f"数值观测 `{key}`：loss 最大绝对差 {value['max_loss_abs_difference']:.6g}；"
                     f"gradient norm 最大绝对差 {value['max_grad_norm_abs_difference']:.6g}。")
    return '\n'.join(lines+['', result['limitation'], ''])
