"""Five-group strict evidence and paired statistics; no GPU imports."""
import math
from pathlib import Path
import statistics

from ablation2_plan import PROTOCOL, POLICIES, definitions, derive
from common import read, require, sha, verify_files, write
from report import fingerprints, paired


def verify_protocol(root, info, offline):
    require(info['bases'] == ['interleaved'] and info['seeds'] == [], 'Wrong five-group scope')
    c = info['config']
    fixed = dict(tp=8, sm_count=132, layers=32, hidden=4096, ffn=16384, heads=32, sequence=2048,
                 micro_batch=1, global_batch=4, seed=1234, token_seed=4321, vocab=32768,
                 warmup_steps=10, timed_steps=20, window=64, fraction=1/64)
    require(all(c[k] == v for k, v in fixed.items()), 'Not the frozen GPT6.7B TP8 protocol')
    seal = read(root/'provenance/preparation.json')
    require(all(seal[k] == info[k] for k in ('protocol', 'run_id', 'source_only', 'executable')) and
            info['executable'] == (not info['source_only']), 'Preparation mode changed; create a new session')
    manifest = read(root/'provenance/ablation-manifest.json')
    require(sha(root/'provenance/ablation-manifest.json') == seal['builds_sha256'] and
            manifest['protocol'] == PROTOCOL and manifest['config'] == c and
            manifest['calibration_hardware'] == info['calibration_hardware'] and
            manifest['candidate_source'] == info['candidate_source'] and
            manifest['parent_plan']['sha256'] == info['parent_plan_sha256'], 'Build registry identity mismatch')
    defs = definitions()
    require(set(info['variants']) == {'interleaved-'+k for k in defs}, 'Missing/extra groups')
    plans = read(root/'scripts/interleaved/selection-plan.json')['entries']
    parent = read(root/'provenance/parent-plan.json')
    require(sha(root/'provenance/parent-plan.json') == info['parent_plan_sha256'], 'Parent plan changed')
    require(set(plans) == set(POLICIES[1:]), 'Native must not have a candidate mask')
    cfg = read(root/'scripts/interleaved/config.json')
    require(cfg['policies'] == list(POLICIES) and cfg['ablation_definitions'] == defs, 'Worker definitions changed')
    for key, definition in defs.items():
        identity = 'interleaved-'+key
        v = info['variants'][identity]
        require(all(v.get(k) == value for k, value in definition.items()), 'Wrong group route')
        if key == 'native_taco':
            require(not any(k in v for k in ('root', 'libraries', 'derived_plan_sha256')), 'Native uses a Flux build')
            continue
        require(plans[key] == derive(parent, definition), 'Candidate physical plan changed')
        m = read(root/'provenance'/f'{identity}-manifest.json')
        mv = m['variants'][identity]
        require(mv['role'] == 'candidate' and mv['sampling'] is False and
                all(mv.get(k) == value for k, value in definition.items()), 'Sampler/wrong candidate build')
        require(v['derived_plan_sha256'] == mv['derived_plan_sha256'] ==
                manifest['variants'][identity]['derived_plan_sha256'] == cfg['derived_plan_sha256'][key], 'Plan hash mismatch')
        if info['executable']:
            require(m['status'] == 'built' and not m.get('diagnostic_only') and len(v['libraries']) == 2,
                    'Unbuilt/diagnostic/source-only session cannot execute')
            if not offline:
                for lib in v['libraries'].values():
                    require(sha(lib['path']) == lib['sha256'], 'Runtime library changed')
    require(plans['G1']['mask'] == plans['G2']['mask'] and not any(map(any, plans['G1']['mask'])), 'G1/G2 must be BF16')
    require(plans['G3']['mask'] == plans['G4']['mask'], 'G3/G4 physical masks differ')
    require(plans['G1']['coords'] == plans['G3']['coords'] and plans['G2']['coords'] == plans['G4']['coords'], 'Arrival pairs differ')


def native_window(record, names, c, mode):
    require(record['flux_library_loaded'] is False and record['native_linear_forward_preserved'] is True and
            not any(k.startswith('libflux') for k in record), 'Native baseline loaded/replaced Flux path')
    mlp = {n for n in names if n.endswith('.mlp.linear_fc2')}
    require(record['codec_placements'] == {n: 'tensor' if n in mlp else 'native' for n in names},
            'Native baseline must retain native attention and tensor MLP communication')
    bf16 = 7*256*4096*2
    expected = dict(mask_sha256=None, wire_bytes=7*256*32*136, bf16_wire_bytes=bf16)
    require(set(record['selection_checks']) == mlp and
            all(v == expected for v in record['selection_checks'].values()), 'Wrong native tensor TACO wire semantics')
    if mode == 'smoke':
        require(set(record['forward_checks']) == mlp, 'Missing native TACO numerical checks')
        for v in record['forward_checks'].values():
            require(v['passed'] and v['budget_vs_bf16'] == .05 and v['budget_vs_codec'] == .01 and
                    math.isfinite(v['max_abs']) and math.isfinite(v['relative_l2']) and
                    math.isfinite(v['protocol_relative_l2']) and
                    v['relative_l2'] <= .05 and v['protocol_relative_l2'] <= .01, 'Native codec numerical budget failed')


def native_profile(events):
    names = {e.get('name') for e in events}
    require({'megatron_taco.encode', 'megatron_taco.exchange', 'megatron_taco.decode_reduce'} <= names,
            'Native tensor TACO encode/exchange/decode profile scopes missing')
    require(any('alltoall' in str(n).lower().replace('_', '') for n in names),
            'Native tensor TACO all-to-all evidence missing')


COMPARISONS = [(k, 'native_taco') for k in POLICIES[1:]] + [
    ('G2', 'G1'), ('G4', 'G3'), ('G3', 'G1'), ('G4', 'G2'), ('G4', 'G1')]


def summarize(values, blocks):
    require(set(values) == set(POLICIES) and all(len(v) == len(blocks) for v in values.values()), 'Incomplete paired blocks')
    require(all(math.isfinite(x) and x > 0 for v in values.values() for x in v), 'Invalid elapsed')
    if not blocks:
        return dict(status='无可用配对块', blocks=0, block_ids=[], comparisons={}, speedups={})
    comparisons = {}
    for group, reference in COMPARISONS:
        logs = [math.log(a/b) for a, b in zip(values[group], values[reference])]
        stats = paired(logs) if len(blocks) >= 2 else dict(percent=100*(1-math.exp(logs[0])), ci95=None, blocks=1)
        comparisons[f'{group}/{reference}'] = dict(stats, log_ratios=logs)
    speedups = {}
    for group in POLICIES[1:]:
        ratios = [b/g for b, g in zip(values['native_taco'], values[group])]
        best = max(range(len(ratios)), key=ratios.__getitem__)
        speedups[group] = dict(median=statistics.median(ratios), ratios=ratios,
                              best=dict(ratio=ratios[best], **blocks[best],
                                        native_taco_ms=values['native_taco'][best], group_ms=values[group][best]))
    return dict(status='ok', blocks=len(blocks), block_ids=blocks, ms_per_step=values,
                median_ms={k: statistics.median(v) for k, v in values.items()},
                median_tokens_per_second={k: statistics.median(8192*1000/x for x in v) for k, v in values.items()},
                comparisons=comparisons, speedups=speedups)


def sensitivity(values, blocks):
    """One non-iterative cutoff per round/policy, one shared whole-block subset."""
    medians = {k: statistics.median(v) for k, v in values.items()}
    excluded = []
    keep = []
    for i, block in enumerate(blocks):
        triggers = {k: dict(ms=values[k][i], median_ms=medians[k], threshold_ms=1.10*medians[k])
                    for k in POLICIES if values[k][i] > 1.10*medians[k]}
        if triggers:
            excluded.append(dict(block, triggers=triggers))
        else:
            keep.append(i)
    selected = {k: [v[i] for i in keep] for k, v in values.items()}
    return dict(original_median_ms=medians, excluded=excluded, remaining_blocks=len(keep),
                summary=summarize(selected, [blocks[i] for i in keep]))


def markdown(result):
    lines = ['# H100 GPT 6.7B TP8 五组消融', '',
             '主结果保留全样本；基线为本会话 native_taco。每窗口取8个rank最大elapsed/20；两轮都是完整训练。',
             'G1相对基线包含共同实现与量化方式的差别，不能解释为交替排序单项收益。',
             '最好加速比是最佳单块观测，不代表稳定收益。95%区间按完整配对块bootstrap（10000次，seed20261009）。',
             '正确性与数据完整性通过；性能增量按区间单独判断，无4%验收门槛。', '']
    for scope, summaries in [('全样本（主结果）', result['rounds']),
                             ('慢窗口敏感性（附表）', {k: v['summary'] for k, v in result['sensitivity'].items()})]:
        lines += [f'## {scope}', '', '| 轮次 | 组 | 中位ms/step | 中位tokens/s |', '| --- | --- | ---: | ---: |']
        for rep, s in summaries.items():
            if not s['blocks']:
                lines.append(f'| {rep} | 无可用配对块 | — | — |')
            for key, ms in s.get('median_ms', {}).items():
                lines.append(f'| {rep} | {key} | {ms:.3f} | {s["median_tokens_per_second"][key]:.2f} |')
        lines += ['', '| 轮次 | 组/native_taco | 配对中位加速比 |', '| --- | --- | ---: |']
        for rep, s in summaries.items():
            for key, speed in s['speedups'].items():
                lines.append(f'| {rep} | {key} | {speed["median"]:.6f} |')
        lines += ['', '| 轮次 | 组 | 最好加速比 | 所在轮/block（0起） | 基线ms | 候选ms |', '| --- | --- | ---: | --- | ---: | ---: |']
        for rep, s in summaries.items():
            for key, speed in s['speedups'].items():
                b = speed['best']
                lines.append(f'| {rep} | {key} | {b["ratio"]:.6f} | {b["round"]}/{b["block"]} | {b["native_taco_ms"]:.3f} | {b["group_ms"]:.3f} |')
        lines += ['', '| 轮次 | 比较 | 配对耗时下降% | 95%区间 | 结论 |', '| --- | --- | ---: | --- | --- |']
        for rep, s in summaries.items():
            for key, stat in s['comparisons'].items():
                ci = stat['ci95']
                interval = f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '不足2块，不估计区间'
                conclusion = '确认正增益' if ci and ci[0] > 0 else '尚未确认正增益'
                lines.append(f'| {rep} | {key} | {stat["percent"]:.3f} | {interval} | {conclusion} |')
        lines.append('')
    for rep, filtered in result['sensitivity'].items():
        lines.append(f'筛选轮次 {rep}：剩余 {filtered["remaining_blocks"]} 块；剔除及触发策略：`{filtered["excluded"]}`。')
    return '\n'.join(lines)+'\n'


def report(root, offline=False):
    from ablation_report import verify_inputs, window, measurement_identity
    root = Path(root)
    info = verify_inputs(root, offline)
    gate = read(root/'preflight.json')
    require(info['executable'] and not info['source_only'] and gate['completed'] and
            gate['protocol'] == PROTOCOL and gate['submission_sha256'] == sha(root/'submission.json'), 'Verified GPU preflight required')
    verify_files(root, gate['files'])
    result = dict(protocol=PROTOCOL, run_id=info['run_id'], completed=False, correctness_passed=True,
                  acceptance_threshold=None, offline=offline, sample_policy='all samples primary',
                  bootstrap=dict(resamples=10000, seed=20261009, unit='complete paired block'),
                  measurement_identity=measurement_identity(info, gate), all_samples=[], rounds={}, sensitivity={})
    combined = {k: [] for k in POLICIES}; combined_blocks = []
    timing_files = {}
    for rep in (1, 2):
        values = {k: [] for k in POLICIES}
        blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == 5 and {j['policy'] for j in jobs} == set(POLICIES), 'Incomplete block')
            for job in jobs:
                directory = root/'results'/job['directory']
                rows = window(root, directory, 'interleaved', job['policy'], 'timing', job['block'], job['index'])
                require(fingerprints(rows) == [tuple(x) for x in gate['fingerprints']], 'Timing initialization/RNG/tokens/software changed')
                elapsed = [r['elapsed_seconds'] for r in rows]
                ms = max(elapsed)*1000/20
                values[job['policy']].append(ms)
                result['all_samples'].append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms, tokens_per_second=8192000/ms))
                timing_files.update({str(p.relative_to(root)): sha(p) for p in directory.rglob('*') if p.is_file()})
        result['rounds'][str(rep)] = summarize(values, blocks)
        result['sensitivity'][str(rep)] = sensitivity(values, blocks)
        for k in POLICIES:
            combined[k] += values[k]
        combined_blocks += blocks
    result['rounds']['combined'] = summarize(combined, combined_blocks)
    require(len(result['all_samples']) == info['timing_windows'], 'Missing timing windows')
    result.update(completed=True, timing_windows=info['timing_windows'], timing_rank_records=8*info['timing_windows'],
                  preflight_rank_records=80, profile_traces=40, candidate_mappings=4,
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=timing_files))
    write(root/'report.json', result)
    (root/'report.md').write_text(markdown(result))
    return result
