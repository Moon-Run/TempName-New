"""CPU-only contracts and statistics for the experimental RTX 5090 TP8 port.

Load this file as ``rtx5090_protocol`` through rtx5090_loader: H100 has an
unrelated module named baseline_protocol whose original contracts stay intact.
"""
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import fmean, mean, median

PROTOCOL = 'rtx5090-phase6-nine-arm-nccl-staged-v2'
TRANSPORT = 'rtx5090-nccl-staged-v2'
TRANSPORT_ENV = {'NCCL_P2P_DISABLE': '1', 'NCCL_SHM_DISABLE': '0', 'NCCL_NVLS_ENABLE': '0'}
ARMS = ('ma11', 'ma89', 'ma8', 'ma9', 'cr3', 'r', 'c', 'a', 'flux-basic')
LABELS = dict(zip(ARMS, ('MA11', 'MA89', 'MA8', 'MA9', 'CR3', 'R', 'C', 'A', 'Flux＋基础量化')))
CR3_FAMILY = ARMS[:5]
QUANT_SELECTIVE = CR3_FAMILY + ('c',)
EQUIVALENCE_GROUPS = (QUANT_SELECTIVE, ('r', 'a'))
POLICY = dict.fromkeys(QUANT_SELECTIVE, 'interleaved_arrival_selective_attn_mlp') | {
    'r': 'interleaved_all_attn_mlp', 'a': 'interleaved_all_attn_mlp', 'flux-basic': 'basic'}
NATIVE_ARM = {'ma11': 'F4-cache-zcount-csoftmax', 'ma89': 'F4-cache-zcount',
              'ma8': 'F4-cache', 'ma9': 'F4-zcount', 'cr3': 'F4',
              'r': 'R', 'c': 'C', 'a': 'A'}
MODEL = dict(layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
             micro_batch=1, global_batch=4, tp=8, dp=1, sm_count=170,
             seed=1234, token_seed=4321, vocab=32768)
TIMING = dict(warmup_steps=10, timed_steps=50, smoke_warmup_steps=2, smoke_steps=3)
TOKENS_PER_STEP = MODEL['sequence'] * MODEL['global_batch']
SEED = 20261030
BLOCKS_PER_CYCLE = 18


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
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def schedule(trials=1, blocks=BLOCKS_PER_CYCLE):
    require(type(trials) is int and trials in (1, 2) and type(blocks) is int and
            blocks >= BLOCKS_PER_CYCLE and blocks % BLOCKS_PER_CYCLE == 0,
            'Use 1/2 trials and a positive multiple of 18 complete blocks')
    # Odd-sized Williams design: every arm occurs twice in each position;
    # every directed adjacent pair occurs twice within the 18 blocks.
    base = (0, 1, 8, 2, 7, 3, 6, 4, 5)
    orders = [tuple(ARMS[(i + k) % len(ARMS)] for i in base) for k in range(len(ARMS))]
    orders += [tuple(reversed(order)) for order in orders[:]]
    result = []
    for trial in range(1, trials + 1):
        rng = random.Random(SEED + trial)
        for cycle in range(blocks // BLOCKS_PER_CYCLE):
            shuffled = list(orders)
            rng.shuffle(shuffled)
            for number, order in enumerate(shuffled):
                block = cycle * BLOCKS_PER_CYCLE + number
                for slot, arm in enumerate(order):
                    result.append(dict(trial=trial, block=block, index=block * len(ARMS) + slot,
                                       arm=arm, mode='timing',
                                       directory=f'compare{trial}/block{block:02d}-{arm}'))
    return result


def jobs(trials=1, blocks=BLOCKS_PER_CYCLE):
    def special(mode):
        return [dict(trial=0, block=-1, index=i, arm=arm, mode=mode,
                     directory=f'{mode}-{arm}') for i, arm in enumerate(ARMS)]
    # Validate the new runtime's GPU path before formal timing. Each window
    # still runs in a fresh process group with its own warmup.
    return special('smoke') + special('profile') + schedule(trials, blocks)


def paired(candidate, reference):
    """Whole-block paired log-ratio bootstrap, matching the H100 estimator."""
    require(len(candidate) == len(reference) >= 6, 'Incomplete paired blocks')
    require(all(math.isfinite(x) and x > 0 for x in (*candidate, *reference)),
            'Paired measurements must be finite and positive')
    logs = [math.log(a / b) for a, b in zip(candidate, reference)]
    rng = random.Random(20261028)
    samples = sorted(100 * (1 - math.exp(fmean(rng.choices(logs, k=len(logs)))))
                     for _ in range(10000))
    return dict(percent=100 * (1 - math.exp(fmean(logs))),
                ci95=[samples[249], samples[9749]], paired_blocks=len(logs))


def summarize(rows, job):
    require(job['mode'] == 'timing' and job['arm'] in ARMS,
            'Only formal timing windows belong in the performance table')
    require(len(rows) == MODEL['tp'] and {r['rank'] for r in rows} == set(range(MODEL['tp'])),
            'All eight distinct rank records are required')
    for row in rows:
        require(row['timed_steps'] == TIMING['timed_steps'] and
                math.isfinite(row['elapsed_seconds']) and row['elapsed_seconds'] > 0,
                'Invalid full-step wall-clock sample')
        for key in ('peak_allocated_gib', 'peak_reserved_gib'):
            require(math.isfinite(row[key]) and row[key] > 0, 'Invalid memory sample')
        require(type(row['native_workspace_bytes']) is int and row['native_workspace_bytes'] >= 0,
                'Missing native CUDA workspace accounting')
    return dict(job, ms_per_step=max(r['elapsed_seconds'] for r in rows) * 1000 / TIMING['timed_steps'],
                peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
                peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),
                native_workspace_gib=max(r['native_workspace_bytes'] for r in rows) / 2**30,
                peak_allocated_plus_taco_workspace_gib=max(
                    r['peak_allocated_gib'] + r['native_workspace_bytes'] / 2**30 for r in rows))


def report(samples, trials=1, blocks=BLOCKS_PER_CYCLE):
    expected = schedule(trials, blocks)
    require(len(samples) == len(expected) and
            all(all(s.get(k) == v for k, v in job.items()) for s, job in zip(samples, expected)),
            'Missing, duplicated or reordered nine-arm timing windows')
    for sample in samples:
        require(all(math.isfinite(sample[k]) and sample[k] > 0 for k in
                    ('ms_per_step', 'peak_allocated_gib', 'peak_reserved_gib')),
                'Invalid timing or memory sample')
        require(math.isfinite(sample['native_workspace_gib']) and sample['native_workspace_gib'] >= 0
                and math.isfinite(sample['peak_allocated_plus_taco_workspace_gib'])
                and sample['peak_allocated_plus_taco_workspace_gib'] >= sample['peak_allocated_gib'],
                'Invalid native/combined memory sample')
    comparisons = {}
    for trial in range(1, trials + 1):
        group = [s for s in samples if s['trial'] == trial]
        values = {arm: [s['ms_per_step'] for s in group if s['arm'] == arm] for arm in ARMS}
        require(all(len(v) == blocks for v in values.values()), 'Incomplete nine-arm trial')
        means = {arm: mean(value) for arm, value in values.items()}
        stats = {arm: dict(mean_ms=means[arm], median_ms=median(values[arm]),
                           min_ms=min(values[arm]), max_ms=max(values[arm]),
                           tokens_per_second=TOKENS_PER_STEP * 1000 / means[arm],
                           ma11_mean_reduction_percent=100 * (1 - means['ma11'] / means[arm]),
                           peak_allocated_gib=mean(s['peak_allocated_gib'] for s in group if s['arm'] == arm),
                           peak_reserved_gib=mean(s['peak_reserved_gib'] for s in group if s['arm'] == arm),
                           max_peak_allocated_gib=max(s['peak_allocated_gib'] for s in group if s['arm'] == arm),
                           max_peak_reserved_gib=max(s['peak_reserved_gib'] for s in group if s['arm'] == arm),
                           max_native_workspace_gib=max(s['native_workspace_gib'] for s in group if s['arm'] == arm),
                           max_peak_allocated_plus_taco_workspace_gib=max(
                               s['peak_allocated_plus_taco_workspace_gib'] for s in group if s['arm'] == arm))
                 for arm in ARMS}
        pairs = {a + '/' + b: paired(values[a], values[b])
                 for i, a in enumerate(ARMS) for b in ARMS[i + 1:]}
        comparisons[f'compare{trial}'] = dict(arms=stats, pairs=pairs)
    return dict(protocol=PROTOCOL, transport=TRANSPORT, original_peer_overlap_preserved=False,
                original_mechanisms_not_identical=True,
                completed=True, trials=trials, blocks=blocks, arms=list(ARMS),
                model=MODEL, timing=TIMING, timing_windows=len(samples),
                timing_rank_records=len(samples) * MODEL['tp'],
                formal_optimizer_steps=len(samples) * TIMING['timed_steps'],
                comparisons=comparisons, all_samples=samples, all_windows_retained=True,
                calibration_origin='frozen_h100_arrival_tables_and_quantization_masks',
                limitation='Experimental non-P2P NCCL staged transport v2; single-node 8×RTX 5090 TP8/DP1, fixed synthetic input. '
                           'Local producer completion precedes packet exchange and local decode. '
                           'Original cross-GPU tile arrival/compute overlap is not preserved. '
                           'R/C/CR3 host submission and buffering now belong to the staged implementation, not the H100 transport. '
                           'Arrival tables and selective masks remain frozen from H100; this is not a 5090 calibration. '
                           'Quantization policies may have different numerical trajectories. '
                           'Each trial is reported separately; all slow windows and negative gains are retained. '
                           'Bootstrap intervals describe this fixed workload and are not adjusted for multiple comparisons.')


def markdown(result):
    require(result['protocol'] == PROTOCOL and result['completed'], 'Incomplete or foreign report')
    lines = ['# RTX 5090 九组 NCCL staging v2 实测汇总', '',
             '本协议不依赖 CUDA peer access；原 H100 跨卡 tile 到达和计算通信重叠不保留，不能与原协议混算。', '',
             '完整 optimizer step；各组均使用全部正式窗口的算术均值。正降幅表示 MA11 更快。', '',
             f"协议：`{PROTOCOL}`；{result['trials']} trial × {result['blocks']} 块 × 9 组；"
             f"共 {result['timing_windows']} 个计时窗口、{result['formal_optimizer_steps']} 个正式 optimizer step。", '']
    for trial, data in result['comparisons'].items():
        lines += [f'## {trial}', '',
                  '| 组别 | 平均 ms/step | 中位数 | 最慢窗口 | tokens/s | MA11 相对均值下降 | 最大 Torch allocated GiB | Torch 峰值＋TACO workspace GiB |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
        for arm in ARMS:
            value = data['arms'][arm]
            lines.append(f"| {LABELS[arm]} | {value['mean_ms']:.3f} | {value['median_ms']:.3f} | "
                         f"{value['max_ms']:.3f} | {value['tokens_per_second']:.2f} | "
                         f"{value['ma11_mean_reduction_percent']:+.3f}% | {value['max_peak_allocated_gib']:.6f} | "
                         f"{value['max_peak_allocated_plus_taco_workspace_gib']:.6f} |")
        lines += ['', '显存合计逐 rank 相加后取最大值，计入独立 cudaMalloc 的 TACO workspace；'
                       '仍不包含 CUDA context、NCCL 等内部显存，不能视为整卡总占用。']
        lines += ['', '以下按完整块的配对对数耗时比估计，使用 10,000 次 bootstrap；与上表均值比是不同统计量。', '',
                  '| 同期配对（前者相对后者） | 配对降幅 %［95% CI］ | 完整块数 |',
                  '| --- | --- | ---: |']
        for pair, value in data['pairs'].items():
            first, second = pair.split('/')
            lines.append(f"| {LABELS[first]} / {LABELS[second]} | {value['percent']:+.3f}"
                         f"［{value['ci95'][0]:+.3f}, {value['ci95'][1]:+.3f}］ | {value['paired_blocks']} |")
        lines.append('')
    return '\n'.join(lines + [result['limitation'], ''])
