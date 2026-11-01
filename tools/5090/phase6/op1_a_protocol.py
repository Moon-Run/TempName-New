"""Independent three-arm full-step experiment; historical nine-arm data stays frozen."""
import itertools
import math
import random
from statistics import mean, median

from rtx5090_loader import load

base = load('protocol')
PROTOCOL = 'rtx5090-phase6-op1-a-direct-wire-v1'
ARMS = ('a', 'flux-basic', 'flux-basic-interleaved')
LABELS = {'a': 'A（5090 direct-wire）', 'flux-basic': 'Flux＋基础量化',
          'flux-basic-interleaved': 'Flux＋基础量化＋交替排序'}
TRANSPORT, TRANSPORT_ENV = base.TRANSPORT, base.TRANSPORT_ENV
MODEL, TIMING, TOKENS_PER_STEP = base.MODEL, base.TIMING, base.TOKENS_PER_STEP
POLICY = {'a': base.POLICY['a'], 'flux-basic': 'basic', 'flux-basic-interleaved': 'basic'}
# Stream-K assigns reduction partitions to logical tiles. Interleaving tiles
# can change rounding; the user explicitly permits this in the OP1 experiment.
EQUIVALENCE_GROUPS = ()
REDUCTION_CHANGES_ALLOWED = True
BLOCKS_PER_CYCLE = 6
read, write, require, sha, digest, paired = base.read, base.write, base.require, base.sha, base.digest, base.paired


def schedule(trials=1, blocks=6):
    require(type(trials) is int and trials in (1, 2) and type(blocks) is int
            and blocks > 0 and blocks % 6 == 0, 'Use 1/2 trials and a positive multiple of 6 blocks')
    result = []
    for trial in range(1, trials + 1):
        rng = random.Random(20261101 + trial)
        for cycle in range(blocks // 6):
            orders = list(itertools.permutations(ARMS))
            rng.shuffle(orders)
            for offset, order in enumerate(orders):
                block = cycle * 6 + offset
                for slot, arm in enumerate(order):
                    result.append(dict(trial=trial, block=block, index=block*3+slot,
                                       arm=arm, mode='timing', directory=f'compare{trial}/block{block:02d}-{arm}'))
    return result


def jobs(trials=1, blocks=6):
    return [dict(trial=0, block=-1, index=i, arm=arm, mode=mode, directory=f'{mode}-{arm}')
            for mode in ('smoke', 'profile') for i, arm in enumerate(ARMS)] + schedule(trials, blocks)


def summarize(rows, job):
    # Shared wall-clock and memory definitions, without treating the sorted arm as historical data.
    require(job['arm'] in ARMS, 'Unknown OP1 arm')
    old_job = dict(job, arm='flux-basic' if job['arm'].startswith('flux') else 'a')
    return dict(base.summarize(rows, old_job), **job)


def report(samples, trials=1, blocks=6):
    expected = schedule(trials, blocks)
    require(len(samples) == len(expected) and all(all(sample.get(k) == v for k, v in job.items())
            for sample, job in zip(samples, expected)), 'Missing, reordered or foreign OP1 timing windows')
    for sample in samples:
        require(all(math.isfinite(sample[k]) and sample[k] > 0 for k in
                    ('ms_per_step', 'peak_allocated_gib', 'peak_reserved_gib')), 'Invalid sample')
        require(math.isfinite(sample['native_workspace_gib']) and sample['native_workspace_gib'] >= 0
                and math.isfinite(sample['peak_allocated_plus_taco_workspace_gib'])
                and sample['peak_allocated_plus_taco_workspace_gib'] >= sample['peak_allocated_gib'],
                'Invalid workspace memory sample')
    comparisons = {}
    for trial in range(1, trials+1):
        group = [row for row in samples if row['trial'] == trial]
        values = {arm: [row['ms_per_step'] for row in group if row['arm'] == arm] for arm in ARMS}
        means = {arm: mean(values[arm]) for arm in ARMS}
        stats = {}
        for arm in ARMS:
            own = [row for row in group if row['arm'] == arm]
            stats[arm] = dict(mean_ms=means[arm], median_ms=median(values[arm]),
                min_ms=min(values[arm]), max_ms=max(values[arm]),
                tokens_per_second=TOKENS_PER_STEP*1000/means[arm],
                a_mean_reduction_percent=100*(1-means['a']/means[arm]),
                max_peak_allocated_gib=max(row['peak_allocated_gib'] for row in own),
                max_peak_reserved_gib=max(row['peak_reserved_gib'] for row in own),
                max_native_workspace_gib=max(row['native_workspace_gib'] for row in own),
                max_peak_allocated_plus_taco_workspace_gib=max(
                    row['peak_allocated_plus_taco_workspace_gib'] for row in own))
        pairs = {a+'/'+b: dict(paired(values[a], values[b]),
                    faster_blocks=sum(x < y for x, y in zip(values[a], values[b])))
                 for a, b in (('a', 'flux-basic'), ('a', 'flux-basic-interleaved'),
                              ('flux-basic-interleaved', 'flux-basic'))}
        comparisons[f'compare{trial}'] = dict(arms=stats, pairs=pairs)
    return dict(protocol=PROTOCOL, transport=TRANSPORT, completed=True, trials=trials, blocks=blocks,
        arms=list(ARMS), model=MODEL, timing=TIMING, timing_windows=len(samples),
        timing_rank_records=len(samples)*8, formal_optimizer_steps=len(samples)*50,
        all_windows_retained=True, all_samples=samples, comparisons=comparisons,
        original_peer_overlap_preserved=False, flux_ordering_equivalence_required=False,
        reduction_changes_allowed=REDUCTION_CHANGES_ALLOWED,
        limitation='Fixed single-node RTX5090 TP8 synthetic workload. No P2P or GEMM/NCCL overlap. '
                   'A and Flux differ in GEMM implementation. The two Flux arms differ in CTA ordering, '
                   'which may also change Stream-K reduction assignment and rounding; exact trajectories are not required. '
                   'All windows, including negative gains, retained; paired bootstrap intervals are '
                   'not adjusted for three comparisons. No historical windows are pooled.')


def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete/foreign OP1 report')
    lines = ['# RTX 5090 OP1-A 三组完整训练对比', '',
        f"协议 `{PROTOCOL}`；{result['trials']} trial × {result['blocks']} blocks × 3 组；"
        f"{result['timing_windows']} 个窗口、{result['formal_optimizer_steps']} 个正式 optimizer step。", '',
        '全部窗口算术均值；每窗口 10 步预热＋50 步正式计时，取八 rank 最大墙钟。', '']
    for trial, data in result['comparisons'].items():
        lines += [f'## {trial}', '',
                  '| 组别 | 平均 ms/step | 中位数 | 最慢窗口 | tokens/s | A 相对均值降幅 | 最大 allocated GiB | allocated＋native GiB |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm, value in data['arms'].items():
            lines.append(f"| {LABELS[arm]} | {value['mean_ms']:.3f} | {value['median_ms']:.3f} | "
                f"{value['max_ms']:.3f} | {value['tokens_per_second']:.2f} | "
                f"{value['a_mean_reduction_percent']:+.3f}% | {value['max_peak_allocated_gib']:.6f} | "
                f"{value['max_peak_allocated_plus_taco_workspace_gib']:.6f} |")
        lines += ['', '| 配对（前者相对后者） | 降幅 % [95% CI] | 更快块数 |', '| --- | ---: | ---: |']
        for name, value in data['pairs'].items():
            a, b = name.split('/')
            lines.append(f"| {LABELS[a]} / {LABELS[b]} | {value['percent']:+.3f} "
                f"[{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}] | "
                f"{value['faster_blocks']}/{value['paired_blocks']} |")
    lines += ['', '正降幅表示前者更快；保留负收益。配对结果与均值比是不同统计量。',
              'allocated＋native 不含 CUDA context/NCCL 内部占用，不能当作整卡峰值。',
              'A 使用不同 GEMM 实现；两个 Flux 组保持原 Stream-K，但排序可能改变归约分配和舍入，不要求逐位同轨迹。']
    if 'flux_numerical_comparison' in result:
        comparison = result['flux_numerical_comparison']
        lines += ['', f"Flux 两组数值观测：{comparison['paired_windows']} 个配对窗口；"
                  f"loss 最大绝对差 {comparison['max_loss_abs_difference']:.6g}，"
                  f"gradient norm 最大绝对差 {comparison['max_grad_norm_abs_difference']:.6g}；"
                  f"最终参数逐位相同的窗口 {comparison['final_parameters_identical_windows']}/{comparison['paired_windows']}。",
                  '这些是同场数值观测，不是模型收敛或质量等价结论。']
    lines += ['', result['limitation'], '']
    return '\n'.join(lines)
