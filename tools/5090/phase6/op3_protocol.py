"""Two concurrent OP3 controls, paired in balanced AB/BA blocks."""
import math
from statistics import mean, median

from rtx5090_loader import load

PROTOCOL = 'rtx5090-phase6-op3-two-arm-v1'
ARMS = ('op3-r', 'op3-improved')
LABELS = dict(zip(ARMS, ('A＋MA89＋MA11＋R＋CR3 消费者融合', 'A＋MA89＋MA11＋OP3 改进')))
POLICY = dict(zip(ARMS, ARMS))
NATIVE = {}
BASIC_ARMS = ARMS
base = load('protocol')
MODEL, TIMING, TOKENS_PER_STEP = base.MODEL, base.TIMING, base.TOKENS_PER_STEP
TRANSPORT, TRANSPORT_ENV = base.TRANSPORT, base.TRANSPORT_ENV
BLOCKS_PER_CYCLE = 2
DEFAULT_BLOCKS = 2
EQUIVALENCE_GROUPS = ()
read, write, require, sha, digest, paired = base.read, base.write, base.require, base.sha, base.digest, base.paired


def definition(arm):
    from op3_sources import definition as describe
    return describe(arm)


def schedule(trials=1, blocks=DEFAULT_BLOCKS):
    require(type(trials) is int and trials in (1, 2) and type(blocks) is int
            and blocks >= 2 and blocks % 2 == 0,
            'Use 1/2 trials and a positive even count of blocks (at least 2)')
    result = []
    for trial in range(1, trials+1):
        for block in range(blocks):
            order = ARMS if (block+trial-1) % 2 == 0 else ARMS[::-1]
            for slot, arm in enumerate(order):
                result.append(dict(trial=trial, block=block, index=block*2+slot, arm=arm,
                    mode='timing', directory=f'compare{trial}/block{block:02d}-{arm}'))
    return result


def jobs(trials=1, blocks=DEFAULT_BLOCKS):
    return [dict(trial=0, block=-1, index=i, arm=arm, mode=mode, directory=f'{mode}-{arm}')
            for mode in ('smoke', 'profile') for i, arm in enumerate(ARMS)] + schedule(trials, blocks)


def summarize(rows, job):
    require(job['arm'] in ARMS, 'Unknown OP3 sample')
    return dict(base.summarize(rows, dict(job, arm='flux-basic')), **job)


def paired_summary(candidate, reference):
    require(len(candidate) == len(reference) and len(candidate) >= 2 and len(candidate) % 2 == 0
            and all(math.isfinite(x) and x > 0 for x in (*candidate, *reference)),
            'Complete positive OP3 block pairs required')
    if len(candidate) >= 6:
        return paired(candidate, reference)
    return dict(percent=100*(1-math.exp(mean(math.log(a/b) for a, b in zip(candidate, reference)))),
                ci95=None, paired_blocks=len(candidate))


def report(samples, trials=1, blocks=DEFAULT_BLOCKS):
    expected = schedule(trials, blocks)
    require(len(samples) == len(expected) and all(all(s.get(k) == v for k, v in job.items())
            for s, job in zip(samples, expected)), 'Missing/reordered/foreign OP3 windows')
    for row in samples:
        require(all(math.isfinite(row[k]) and row[k] > 0 for k in
                    ('ms_per_step', 'peak_allocated_gib', 'peak_reserved_gib')), 'Invalid sample')
        require(math.isfinite(row['native_workspace_gib']) and row['native_workspace_gib'] >= 0
                and math.isfinite(row['peak_allocated_plus_taco_workspace_gib'])
                and row['peak_allocated_plus_taco_workspace_gib'] >= row['peak_allocated_gib'], 'Invalid memory')
    comparisons = {}
    reference, candidate = ARMS
    for trial in range(1, trials+1):
        own = [row for row in samples if row['trial'] == trial]
        values = {arm: [row['ms_per_step'] for row in own if row['arm'] == arm] for arm in ARMS}
        means = {arm: mean(values[arm]) for arm in ARMS}
        stats = {arm: dict(mean_ms=means[arm], median_ms=median(values[arm]), min_ms=min(values[arm]),
            max_ms=max(values[arm]), tokens_per_second=TOKENS_PER_STEP*1000/means[arm],
            reduction_vs_r_percent=100*(1-means[arm]/means[reference]),
            **{'max_'+key: max(row[key] for row in own if row['arm'] == arm) for key in
               ('peak_allocated_gib', 'peak_reserved_gib', 'native_workspace_gib',
                'peak_allocated_plus_taco_workspace_gib')}) for arm in ARMS}
        pairs = {candidate+'/'+reference: dict(paired_summary(values[candidate], values[reference]),
                    faster_blocks=sum(x < y for x, y in zip(values[candidate], values[reference])))}
        comparisons[f'compare{trial}'] = dict(arms=stats, pairs=pairs)
    return dict(protocol=PROTOCOL, transport=TRANSPORT, completed=True, arms=list(ARMS),
        definitions={arm: definition(arm) for arm in ARMS}, trials=trials, blocks=blocks,
        counterbalanced_order=True, pairwise_ci_available=blocks >= 6,
        model=MODEL, timing=TIMING, timing_windows=len(samples), timing_rank_records=len(samples)*8,
        formal_optimizer_steps=len(samples)*50, all_windows_retained=True, comparisons=comparisons,
        all_samples=samples, numerical_equivalence_required=False, reduction_changes_allowed=False,
        raw_projection_bitwise_required=True, smoke_training_equivalence_required=True,
        full_training_equivalence_required=False,
        original_peer_overlap_preserved=False,
        mechanism_stack=['ma8_activation_cache', 'ma9_zero_count', 'ma11_softmax'],
        increment_over_unoptimized_validated=False,
        limitation='Both arms share the same A projection foundation and one installation of MA8, MA9 and MA11. '
                   'The registered per-arm definitions identify the R and OP3 changes. '
                   'AB/BA order is balanced within every two blocks. All windows and negative gains are retained. '
                   'This comparison estimates the difference between these two concurrent arms; it does not isolate '
                   'the incremental gain over the historical OP2 baseline. No historical pooling. '
                   + (f'{blocks} paired blocks provide a descriptive point estimate only; no confidence interval.'
                      if blocks < 6 else 'The paired interval uses 10,000 bootstrap samples.'))


def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete/foreign OP3 report')
    lines = ['# RTX 5090 OP3：R 方案与实测瓶颈改进方案', '', f"协议 `{PROTOCOL}`；{result['trials']} trial × "
        f"{result['blocks']} blocks × 2 组，共 {result['timing_windows']} 窗口、"
        f"{result['formal_optimizer_steps']} 个正式 optimizer step。", '',
        '每两块采用 AB/BA 顺序；每个窗口独立启动，均保留 10 个 warmup 与 50 个正式 step。', '']
    if result['blocks'] < 6:
        lines += [f"{result['blocks']} 块短测已完成排序平衡；只报告描述统计与配对点估计，不计算 95% CI。", '']
    for trial, record in result['comparisons'].items():
        lines += [f'## {trial}', '',
            '| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 相对 R 方案降幅 | 最大 allocated GiB | allocated＋native GiB |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            v = record['arms'][arm]
            lines.append(f"| {LABELS[arm]} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | "
                f"{v['max_ms']:.3f} | {v['tokens_per_second']:.2f} | {v['reduction_vs_r_percent']:+.3f}% | "
                f"{v['max_peak_allocated_gib']:.6f} | {v['max_peak_allocated_plus_taco_workspace_gib']:.6f} |")
        lines += ['', '| 配对（改进方案 / R 方案） | 降幅 % [95% CI] | 更快块数 |', '| --- | ---: | ---: |']
        for key, v in record['pairs'].items():
            a, b = key.split('/')
            interval = (f"[{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]" if v['ci95'] is not None
                        else f"［{result['blocks']} 块短测，不计算 CI］")
            lines.append(f"| {LABELS[a]} / {LABELS[b]} | {v['percent']:+.3f} "
                f"{interval} | {v['faster_blocks']}/{v['paired_blocks']} |")
    lines += ['', '均值包含全部窗口；配对估计使用完整块的对数比。正值表示改进方案比 R 方案更快，保留负收益。',
              'allocated＋native 不含 CUDA context、NCCL 等内部占用；逻辑贡献字节数不是 NCCL 实测链路流量。',
              '两组裸投影的 packet 与输出必须通过逐位一致检查，源归约顺序不允许改变；smoke 的 loss、gradient norm、全部参数梯度与更新轨迹必须一致，随后计时窗口的训练轨迹作观测。',
              '模型 smoke 中 R 方案记录融合 bias/residual 后的输出，改进方案记录裸投影；这两类输出哈希不直接比较。']
    for key, value in result.get('numerical_comparisons', {}).items():
        lines.append(f"数值观测 `{key}`：loss 最大绝对差 {value['max_loss_abs_difference']:.6g}；"
                     f"gradient norm 最大绝对差 {value['max_grad_norm_abs_difference']:.6g}。")
    return '\n'.join(lines+['', result['limitation'], ''])
