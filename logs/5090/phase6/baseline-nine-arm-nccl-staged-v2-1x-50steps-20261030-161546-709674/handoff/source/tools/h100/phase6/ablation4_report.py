"""Strict five-policy, dual-role gates and complete-block timing comparisons."""
import csv
import gzip
import json
from pathlib import Path
import statistics

import ablation3_report
import ablation4_build as builder
import ablation4_protocol as routing
import ablation4_runtime as communication
from ablation4_prepare import role_hashes
from ablation_report import measurement_identity
from common import read, require, sha, verify_files, write
from report import fingerprints
from scope_plan import model_config, hardware_identity
import scope_report
import ablation_timing

PROTOCOL, POLICIES = routing.PROTOCOL, routing.POLICIES


def verify_inputs(root, offline=False):
    root = Path(root); info = read(root/'submission.json')
    require(info['protocol'] == PROTOCOL and info['config'] == model_config(), 'Wrong ablation4 protocol/model')
    verify_files(root, info['files'])
    expected = routing.schedule(info['blocks_per_round'], info['schedule_seed'], info['rounds'])
    require(info['schedule'] == read(root/'schedule.json') == expected and info['timing_windows'] == len(expected), 'Five-policy schedule changed')
    seal = read(root/'provenance/preparation.json'); cfg = read(root/'scripts/config.json')
    require(all(seal[k] == info[k] for k in ('protocol', 'run_id', 'source_only', 'executable', 'rounds', 'communication')) and
            info['executable'] == (not info['source_only']), 'Rehearsal/round/communication identity changed')
    require(cfg['protocol'] == PROTOCOL and cfg['policies'] == list(POLICIES) and cfg['repetitions'] == info['rounds'] and
            cfg['communication'] == communication.validate(info['communication']), 'Worker configuration changed')
    ablation_timing.validate(root, info)
    require(cfg['graph_forward'] is False and
            cfg['double_buffered'] is True and cfg['ring_reduction'] is True, 'Full optimizer-step protocol changed')
    model = model_config()
    require(cfg['common'] == {k: model[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')} and
            cfg['cases'] == [dict(name='h100-phase6', **{k: model[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
            'Worker model/full shapes changed')
    plans = read(root/'scripts/selection-plan.json'); m = read(root/'provenance/build-manifest.json')
    candidate = read(root/'provenance/candidate-manifest.json')
    fused = read(root/'provenance/fused-manifest.json')
    require(sha(root/'scripts/selection-plan.json') == seal['plans_sha256'] == m['files']['plans.json'] and
            cfg['role_plan_sha256'] == role_hashes(plans), 'Shared role plans changed')
    require(sha(root/'provenance/build-manifest.json') == info['build_manifest']['sha256'] and
            sha(root/'provenance/candidate-manifest.json') == m['candidate_manifest_sha256'], 'Build snapshots changed')
    builder.validate_snapshots(m, plans, candidate, fused, info['executable'], verify_evidence=not offline)
    if info['executable']:
        require(all(hardware_identity(r['hardware']) == hardware_identity(info['calibration_hardware'])
                    for r in plans['parent']['roles'].values()), 'Calibrated hardware/order changed')
    require(set(info['variants']) == set(POLICIES), 'Missing/extra ablation4 policies')
    checked = set(); build_root = Path(info['build_manifest']['path']).parent
    for p in POLICIES:
        v, d = info['variants'][p], routing.definition(p)
        require(all(v.get(k) == value for k, value in d.items()), 'Policy implementation/role definition changed')
        if routing.is_native(p):
            require(not any(k in v for k in ('root', 'libraries', 'sampling')), 'Native baseline must not load Flux')
            continue
        require(Path(v['root']) == builder.runtime_root(build_root, p) and v['sampling'] is False, 'Wrong policy build/order/sampling')
        if info['executable']:
            require(set(v['libraries']) == {'libflux_cuda.so', 'libflux_cuda_ths_op.so'}, 'Missing runtime libraries')
            for lib in v['libraries'].values():
                key = (lib['path'], lib['sha256'])
                if not offline and key not in checked:
                    require(sha(lib['path']) == lib['sha256'], 'Runtime library changed'); checked.add(key)
    if not offline:
        verify_files('/', info['build_files'])
        require(sha(info['build_manifest']['path']) == info['build_manifest']['sha256'], 'Build manifest changed')
    return info


def window(root, folder, policy, mode, block=-1, index=-1):
    rows = scope_report.window(root, folder, policy, mode, block, index, routing)
    info = read(Path(root)/'submission.json'); command = read(Path(folder)/'command.json')
    require(command['communication'] == info['communication'] and command['scope'] == routing.definition(policy) and
            command['policy'] == policy and command['mode'] == mode and command['block'] == block and command['index'] == index,
            'Command communication/policy identity changed')
    require(all(r.get('communication') == info['communication'] for r in rows), 'Rank communication environment changed')
    return rows


def profile_roles(events, policy):
    return scope_report.profile_roles(events, policy, routing)


def verify_mapping(rows, policy, role, plans):
    require(len(rows) == 4096 and {r['rank'] for r in rows} == set(range(8)), 'Incomplete GPU mapping')
    for rank in range(8):
        rr = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
        require([r['index'] for r in rr] == list(range(512)) and
                sorted(r['m']*32+r['n'] for r in rr) == list(range(512)) and
                all(r['destination'] == r['m']//2 and r['sm_occupancy'] > 0 for r in rr), 'Invalid mapping coverage')
        if routing.is_original(policy):
            require(sorted(r['base_m']*32+r['base_n'] for r in rr) == list(range(512)) and
                    all(r['m'] == (r['base_m']+2*rank)%16 and r['n'] == r['base_n'] for r in rr),
                    'Fused-v2 must preserve original Flux rank-offset order')
        else:
            require([r['m']*32+r['n'] for r in rr] == routing.role_plans(plans, policy)[role]['entry']['coords'][rank],
                    'Wrong candidate role/order mapping')


def preflight(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline)
    require(info['executable'] and not info['source_only'], 'CPU rehearsal cannot certify GPU execution')
    start = read(root/'start-gpu-check.json')
    require(start['passed'] and start['run_id'] == info['run_id'], 'Missing startup check')
    files = [root/'start-gpu-check.json']; reference = None; profiles = {}
    plans = read(root/'scripts/selection-plan.json')
    for p in POLICIES:
        for mode in ('smoke', 'profile'):
            folder = root/'results'/f'{mode}-{p}'; rows = window(root, folder, p, mode)
            key = fingerprints(rows)
            if reference is None: reference = key
            require(key == reference, 'Cross-policy initialization/RNG/tokens/software mismatch')
            files += [f for f in folder.rglob('*') if f.is_file()]
        for rank in range(8):
            with gzip.open(root/'results'/f'profile-{p}'/f'rank{rank}-profile.json.gz', 'rt') as f:
                profiles[f'{p}/rank{rank}'] = profile_roles(json.load(f)['traceEvents'], p)
        if not routing.is_native(p):
            for role in routing.SHAPES:
                path = root/'results'/f'mapping-{p}-{role}.csv'
                with path.open() as f:
                    verify_mapping([{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)], p, role, plans)
                files.append(path)
    gate = dict(protocol=PROTOCOL, completed=True, submission_sha256=sha(root/'submission.json'),
                communication=info['communication'], fingerprints=reference, role_profiles=profiles,
                files={str(p.relative_to(root)): sha(p) for p in files})
    write(root/'preflight.json', gate)
    return gate


def with_means(summary):
    if summary['blocks']:
        summary['mean_ms'] = {p: statistics.mean(v) for p, v in summary['ms_per_step'].items()}
        summary['tokens_per_second'] = {p: 8192000/x for p, x in summary['mean_ms'].items()}
    return summary


def summarize(values, blocks):
    return with_means(ablation3_report.summarize(values, blocks, POLICIES, routing.COMPARISONS))


def sensitivity(values, blocks):
    result = ablation3_report.sensitivity(values, blocks, POLICIES, routing.COMPARISONS)
    with_means(result['summary'])
    return result


def markdown(result):
    lines = ['# H100 GPT6.7B TP8 消融实验4：五组 MLP＋Attention', '',
             f'本次 {result["round_count"]} 轮，每轮 {result["blocks_per_round"]} 个完整五组块，'
             f'共 {result["timing_windows"]} 个正式窗口。所有组均启用64个输出投影。',
             f'每窗口预热{result["execution_timing"]["warmup_steps"]}步，正式计时{result["execution_timing"]["timed_steps"]}步；'
             f'合计{result["formal_optimizer_steps"]}个正式optimizer step（不重复计rank）。',
             '计时是完整32层模型、4微批的optimizer step，8192 token/step；全样本主表，不筛选慢窗口。',
             '完整量化仅指全部远端贡献；本地贡献、GEMM及反向保持既定BF16/STE路径。',
             'Flux/A/B共用固定融合v2 codec、逐次配置/复位、单工作区和输出clone，只改变调度顺序；C沿用消融3双缓冲实现。',
             '因此A/Flux、B/A分别比较交替和到达优先；C/B同时包含选择性量化与实现优化，不是仅切换mask。',
             '配对95%区间按完整五组块bootstrap10000次、seed20261009；无4%验收门槛、无多重比较校正。',
             '通信配置：`'+json.dumps(result['communication'], ensure_ascii=False, sort_keys=True)+'`', '']
    if result['round_count'] == 1:
        lines += ['单轮没有第二轮反转验证；combined与第1轮是同一批样本。', '']
    for label, summaries in [('全样本', result['rounds']), ('慢窗口敏感性', {k: v['summary'] for k, v in result['sensitivity'].items()})]:
        lines += ['## '+label, '', '| 轮次 | 策略 | 平均 ms/step | 中位 ms/step | tokens/s（总量/总时间） |',
                  '| --- | --- | ---: | ---: | ---: |']
        for rep, s in summaries.items():
            if not s['blocks']:
                lines.append(f'| {rep} | 无可用完整块 | — | — | — |')
            for p, ms in s.get('mean_ms', {}).items():
                lines.append(f'| {rep} | {routing.LABELS[p]} | {ms:.3f} | {s["median_ms"][p]:.3f} | {s["tokens_per_second"][p]:.2f} |')
        lines += ['', '| 轮次 | X / 参考Y | 配对下降% | 95% CI | 中位加速比 | 最好单块 |',
                  '| --- | --- | ---: | --- | ---: | ---: |']
        for rep, s in summaries.items():
            for key, v in s['comparisons'].items():
                a, b = key.split('/'); ci = v['ci95']; speed = s['speedups'][key]
                text = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '不足两块'
                lines.append(f'| {rep} | {routing.LABELS[a]} / {routing.LABELS[b]} | {v["percent"]:+.3f} | '
                             f'{text} | {speed["median"]:.6f} | {speed["best"]["ratio"]:.6f} |')
    lines += ['', '最好值仅为单块观测；完整块配对才可比较，负收益和跨零区间保留。']
    for rep, s in result['sensitivity'].items():
        lines.append(f'轮次 {rep}：保留 {s["remaining_blocks"]} 块，排除与触发记录：`{s["excluded"]}`。')
    return '\n'.join(lines)+'\n'


def report(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline); gate = read(root/'preflight.json')
    timing = ablation_timing.validate(root, info)
    require(info['executable'] and not info['source_only'] and gate['completed'] and gate['protocol'] == PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json') and gate['communication'] == info['communication'],
            'Completed GPU preflight under the same communication settings required')
    verify_files(root, gate['files'])
    observed = {str(p.relative_to(root/'results')) for p in (root/'results').glob('round*/window-*') if p.is_dir()}
    require(observed == {j['directory'] for j in info['schedule']}, 'Missing or extra timing window directories')
    result = dict(protocol=PROTOCOL, run_id=info['run_id'], completed=False, offline=offline, all_samples=[],
                  rounds={}, sensitivity={}, round_count=info['rounds'], blocks_per_round=info['blocks_per_round'],
                  communication=info['communication'], execution_timing=timing,
                  formal_optimizer_steps=info['timing_windows']*timing['timed_steps'],
                  acceptance_threshold=None, sample_policy='all samples primary',
                  correctness_passed=True, bootstrap=dict(resamples=10000, seed=20261009, unit='complete five-policy block',
                                                       multiple_comparison_correction=None),
                  primary_comparisons=routing.PRIMARY, comparisons=routing.COMPARISONS, config=info['config'],
                  candidate_source_commit=info['candidate_source_commit'], measurement_identity=measurement_identity(info, gate),
                  role_plan_sha256=read(root/'scripts/config.json')['role_plan_sha256'])
    result['measurement_identity']['communication'] = info['communication']
    combined = {p: [] for p in POLICIES}; combined_blocks = []; files = {}
    for rep in range(1, info['rounds']+1):
        values = {p: [] for p in POLICIES}; blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == 5 and {j['policy'] for j in jobs} == set(POLICIES), 'Incomplete five-policy block')
            for job in jobs:
                folder = root/'results'/job['directory']; rows = window(root, folder, job['policy'], 'timing', job['block'], job['index'])
                require(fingerprints(rows) == [tuple(x) for x in gate['fingerprints']], 'Timing initialization/RNG/tokens/software changed')
                elapsed = [r['elapsed_seconds'] for r in rows]; ms = max(elapsed)*1000/timing['timed_steps']
                values[job['policy']].append(ms)
                result['all_samples'].append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms, tokens_per_second=8192000/ms))
                files.update({str(p.relative_to(root)): sha(p) for p in folder.rglob('*') if p.is_file()})
        result['rounds'][str(rep)] = summarize(values, blocks)
        result['sensitivity'][str(rep)] = sensitivity(values, blocks)
        for p in POLICIES: combined[p] += values[p]
        combined_blocks += blocks
    result['rounds']['combined'] = summarize(combined, combined_blocks)
    require(len(result['all_samples']) == info['timing_windows'], 'Missing timing windows')
    result.update(completed=True, timing_windows=info['timing_windows'], timing_rank_records=8*info['timing_windows'],
                  preflight_rank_records=80, profile_traces=40, candidate_shape_mappings=8,
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=files))
    write(root/'report.json', result); (root/'report.md').write_text(markdown(result))
    return result
