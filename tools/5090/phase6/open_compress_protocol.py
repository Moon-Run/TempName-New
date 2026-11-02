"""Independent three-arm protocol; a one-block run is explicitly exploratory."""
import itertools
import math
import random
from statistics import mean, median
from rtx5090_loader import load

base = load('protocol')
PROTOCOL = 'rtx5090-open-compress-int8-three-arm-v1'
ARMS = ('flux-bf16', 'flux-int8', 'flux-int8-interleaved')
LABELS = dict(zip(ARMS, ('Flux（BF16）', 'Flux＋block INT8', 'Flux＋block INT8＋交替排序')))
POLICY = {arm: arm for arm in ARMS}
MODEL, TIMING, TOKENS_PER_STEP = base.MODEL, base.TIMING, base.TOKENS_PER_STEP
TRANSPORT, TRANSPORT_ENV = base.TRANSPORT, base.TRANSPORT_ENV
read, write, require, sha, digest = base.read, base.write, base.require, base.sha, base.digest
PACKET_BYTES = 256 * 4096 // 128 * 132


def schedule(trials=1, blocks=6):
    require(type(trials) is int and trials > 0 and type(blocks) is int and blocks > 0,
            'trials and blocks must be positive integers')
    result = []
    for trial in range(1, trials + 1):
        rng = random.Random(20261102 + trial)
        for cycle in range((blocks + 5) // 6):
            orders = list(itertools.permutations(ARMS))
            rng.shuffle(orders)
            for offset, order in enumerate(orders):
                block = cycle * 6 + offset
                if block >= blocks:
                    break
                for slot, arm in enumerate(order):
                    result.append(dict(trial=trial, block=block, index=block*3+slot, arm=arm,
                        mode='timing', directory=f'compare{trial}/block{block:02d}-{arm}'))
    return result


def jobs(trials=1, blocks=6):
    return [dict(trial=0, block=-1, index=i, arm=arm, mode=mode, directory=f'{mode}-{arm}')
            for mode in ('smoke', 'profile') for i, arm in enumerate(ARMS)] + schedule(trials, blocks)


def summarize(rows, job):
    require(job['arm'] in ARMS, 'Unknown arm')
    value = base.summarize(rows, dict(job, arm='flux-basic'))
    value['peak_allocated_plus_native_gib'] = value.pop('peak_allocated_plus_taco_workspace_gib')
    return dict(value, **job)


def report(samples, trials=1, blocks=6):
    expected = schedule(trials, blocks)
    require(len(samples) == len(expected) and all(all(s.get(k) == v for k, v in j.items())
            for s, j in zip(samples, expected)), 'Missing, reordered, duplicate or foreign windows')
    for s in samples:
        require(all(math.isfinite(s[k]) and s[k] > 0 for k in
                ('ms_per_step', 'peak_allocated_gib', 'peak_reserved_gib', 'peak_allocated_plus_native_gib'))
                and s['native_workspace_gib'] == 0, 'Invalid measurement or unexpected native allocation')
    comparisons = {}
    for trial in range(1, trials+1):
        own = [s for s in samples if s['trial'] == trial]
        values = {a: [s['ms_per_step'] for s in own if s['arm'] == a] for a in ARMS}
        stats = {a: dict(mean_ms=mean(values[a]), median_ms=median(values[a]),
            min_ms=min(values[a]), max_ms=max(values[a]), tokens_per_second=TOKENS_PER_STEP*1000/mean(values[a]),
            reduction_vs_bf16_percent=100*(1-mean(values[a])/mean(values['flux-bf16'])),
            **{'max_'+k: max(s[k] for s in own if s['arm'] == a) for k in
               ('peak_allocated_gib', 'peak_reserved_gib', 'peak_allocated_plus_native_gib')}) for a in ARMS}
        pairs = {}
        for a, b in ((ARMS[1], ARMS[0]), (ARMS[2], ARMS[1]), (ARMS[2], ARMS[0])):
            item = (base.paired(values[a], values[b]) if blocks >= 6 else
                dict(percent=100*(1-math.exp(mean(math.log(x/y) for x,y in zip(values[a],values[b])))),
                     ci95=None, paired_blocks=blocks))
            pairs[a+'/'+b] = dict(item, mean_reduction_percent=100*(1-mean(values[a])/mean(values[b])),
                                 faster_blocks=sum(x < y for x,y in zip(values[a],values[b])))
        comparisons[f'compare{trial}'] = dict(arms=stats, pairs=pairs)
    return dict(protocol=PROTOCOL, completed=True, trials=trials, blocks=blocks, arms=list(ARMS),
        model=MODEL, timing=TIMING, comparisons=comparisons, all_samples=samples,
        timing_windows=len(samples), timing_rank_records=8*len(samples), formal_optimizer_steps=50*len(samples),
        balanced_order=blocks % 6 == 0, exploratory=blocks < 6, all_windows_retained=True,
        arrival=False, selective=False, original_peer_overlap_preserved=False,
        limitation='Fixed synthetic GPT6.7B workload. BF16 uses NCCL ReduceScatter; INT8 uses byte AllToAll '
          'and source-ordered BF16 reduction. INT8/BF16 is a whole-path comparison. Interleaving changes '
          'GEMM CTA mapping and can reassign StreamK reductions; no bitwise trajectory equivalence is claimed. '
          'No tile-arrival overlap, arrival policy, selective quantization or convergence claim. '
          'Payload bytes are not measured NCCL link traffic. CI only with >=6 blocks; comparisons unadjusted.')


def markdown(result):
    require(result['completed'] and result['protocol'] == PROTOCOL, 'Incomplete report')
    lines = ['# RTX 5090 开源通信压缩三组对比', '', f"协议 `{PROTOCOL}`；"
             f"{result['trials']} trials × {result['blocks']} blocks；{result['formal_optimizer_steps']} 正式 optimizer steps。",
             '', '到达优先、选择性量化均关闭；全部计时窗口保留。']
    if result['exploratory']:
        lines += ['', '**单批探索性结果：不足六个完整块，不生成置信区间。**']
    for trial, group in result['comparisons'].items():
        lines += ['', f'## {trial}', '', '| 组别 | 平均 ms/step | 中位数 | 最慢 | tokens/s | 最大 allocated GiB | 最大 reserved GiB |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            s = group['arms'][arm]
            lines.append(f"| {LABELS[arm]} | {s['mean_ms']:.3f} | {s['median_ms']:.3f} | {s['max_ms']:.3f} | {s['tokens_per_second']:.2f} | "
                         f"{s['max_peak_allocated_gib']:.4f} | {s['max_peak_reserved_gib']:.4f} |")
        lines += ['', '| 配对（前者/后者） | 均值比降幅 % | 配对对数比降幅 % | 配对 95% CI | 更快块 |',
                  '| --- | ---: | ---: | --- | ---: |']
        for key, s in group['pairs'].items():
            ci = '不估计' if s['ci95'] is None else f"[{s['ci95'][0]:+.3f}, {s['ci95'][1]:+.3f}]"
            lines.append(f"| {key} | {s['mean_reduction_percent']:+.3f} | {s['percent']:+.3f} | {ci} | {s['faster_blocks']}/{s['paired_blocks']} |")
    lines += ['', '算子、分段 CUDA event 与同步 wall time、显存见 `operator-summary.json`；原始证据见 `operator/`。',
              'Torch allocated/reserved 分开报告；codec 无外部 cudaMalloc；整卡峰值仍不含 NCCL/context 的精确归因。',
              '', result['limitation'], '']
    return '\n'.join(lines)
