"""OP3 combined encoder + R + consumer fusion; one measured configuration."""
import math
from statistics import mean, median

from rtx5090_loader import load

PROTOCOL = 'rtx5090-phase6-op3-combined-v2'
COMBINED = 'op3-combined'
ARMS = (COMBINED,)
# Retain the two component implementations for independent correctness tests.
ALL_ARMS = ('op3-r', 'op3-improved', COMBINED)
LABELS = {COMBINED: 'A＋MA89＋MA11＋R＋CR3 消费者融合＋OP3 编码器改进'}
POLICY = dict(zip(ALL_ARMS, ALL_ARMS))
NATIVE = {}
BASIC_ARMS = ALL_ARMS
base = load('protocol')
MODEL, TIMING, TOKENS_PER_STEP = base.MODEL, base.TIMING, base.TOKENS_PER_STEP
TRANSPORT, TRANSPORT_ENV = base.TRANSPORT, base.TRANSPORT_ENV
DEFAULT_BLOCKS = 1
read, write, require, sha, digest = base.read, base.write, base.require, base.sha, base.digest


def fused(arm):
    require(arm in ALL_ARMS, 'Unknown OP3 arm')
    return arm != 'op3-improved'


def build_arm(arm):
    require(arm in ALL_ARMS, 'Unknown OP3 arm')
    # All kernels already exist in the immutable v1 binary; do not rewrite its
    # manifest or recipe to add a Python composition of two existing exports.
    return 'op3-r' if arm == COMBINED else arm


def definition(arm):
    from op3_sources import definition as describe
    return describe(arm)


def schedule(trials=1, blocks=DEFAULT_BLOCKS):
    require(type(trials) is int and trials in (1, 2) and type(blocks) is int and blocks > 0,
            'Use 1/2 trials and a positive integer block count')
    return [dict(trial=trial, block=block, index=block, arm=COMBINED, mode='timing',
                 directory=f'compare{trial}/block{block:02d}-{COMBINED}')
            for trial in range(1, trials+1) for block in range(blocks)]


def jobs(trials=1, blocks=DEFAULT_BLOCKS):
    return [dict(trial=0, block=-1, index=0, arm=COMBINED, mode=mode, directory=f'{mode}-{COMBINED}')
            for mode in ('smoke', 'profile')] + schedule(trials, blocks)


def summarize(rows, job):
    require(job['arm'] in ARMS, 'Unknown OP3 measured arm')
    return dict(base.summarize(rows, dict(job, arm='flux-basic')), **job)


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
    for trial in range(1, trials+1):
        own = [row for row in samples if row['trial'] == trial]
        values = [row['ms_per_step'] for row in own]
        stats = dict(mean_ms=mean(values), median_ms=median(values), min_ms=min(values), max_ms=max(values),
            tokens_per_second=TOKENS_PER_STEP*1000/mean(values),
            **{'max_'+key: max(row[key] for row in own) for key in
               ('peak_allocated_gib', 'peak_reserved_gib', 'native_workspace_gib',
                'peak_allocated_plus_taco_workspace_gib')})
        comparisons[f'compare{trial}'] = dict(arms={COMBINED: stats}, pairs={})
    return dict(protocol=PROTOCOL, transport=TRANSPORT, completed=True, arms=list(ARMS),
        definitions={COMBINED: definition(COMBINED)}, trials=trials, blocks=blocks,
        counterbalanced_order=False, pairwise_ci_available=False,
        model=MODEL, timing=TIMING, timing_windows=len(samples), timing_rank_records=len(samples)*8,
        formal_optimizer_steps=len(samples)*50, all_windows_retained=True, comparisons=comparisons,
        all_samples=samples, reduction_changes_allowed=False, raw_projection_bitwise_required=True,
        operator_gradient_equivalence_required=True, full_training_cross_arm_comparison=False,
        original_peer_overlap_preserved=False,
        mechanism_stack=['ma8_activation_cache', 'ma9_zero_count', 'ma11_softmax',
                         'r_prefetch_decoder', 'cr3_consumer_fusion', 'warp_owned_encoder'],
        increment_over_unoptimized_validated=False,
        limitation='Single combined configuration. All windows are retained. Operator forward/gradient '
                   'equivalence is checked against the unfused autograd reference before measurement. '
                   'No concurrent performance control, paired gain, paired confidence interval or historical pooling.')


def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete/foreign OP3 report')
    lines = ['# RTX 5090 OP3：编码器＋R＋CR3 融合组合组', '',
        f"协议 `{PROTOCOL}`；{result['trials']} trial × {result['blocks']} blocks × 1 组，共 "
        f"{result['timing_windows']} 窗口、{result['formal_optimizer_steps']} 个正式 optimizer step。", '',
        '每窗口独立启动，10 个 warmup＋50 个正式 step；只测组合组，不计算同期相对降幅或配对 CI。', '',
        '| trial | 平均 ms/step | 中位数 | 最快 | 最慢 | tokens/s | 最大 allocated GiB | allocated＋native GiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for trial, record in result['comparisons'].items():
        v = record['arms'][COMBINED]
        lines.append(f"| {trial} | {v['mean_ms']:.3f} | {v['median_ms']:.3f} | {v['min_ms']:.3f} | "
            f"{v['max_ms']:.3f} | {v['tokens_per_second']:.2f} | {v['max_peak_allocated_gib']:.6f} | "
            f"{v['max_peak_allocated_plus_taco_workspace_gib']:.6f} |")
    lines += ['', '算子延迟及阶段 profile 见 `operator/report.md`；算子时间不等于整步耗时。',
              'allocated＋native 不含 CUDA context、NCCL 等内部占用。', '', result['limitation'], '']
    return '\n'.join(lines)
