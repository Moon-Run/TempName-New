"""Nine-group admission, role-aware preflight and 18 explicitly paired comparisons."""
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

import ablation3_protocol as routing
import ablation3_build as builder
import scope_report
from ablation3_protocol import PROTOCOL, POLICIES, PRIMARY, RANGE_COMPARISONS, COMPARISONS, definition, schedule
from ablation_report import measurement_identity
from common import read, require, sha, verify_files, write
from report import fingerprints, paired
from scope_plan import model_config, hardware_identity


def verify_inputs(root, offline=False):
    root = Path(root); info = read(root/'submission.json')
    require(info['protocol'] == PROTOCOL and info['config'] == model_config(), 'Wrong ablation3 protocol/model')
    verify_files(root, info['files'])
    # Older sessions always froze two rounds, even when their measurements stopped early.
    rounds = info.get('rounds', 2)
    require(info['schedule'] == read(root/'schedule.json') == schedule(info['blocks_per_round'], info['schedule_seed'], rounds) and
            info['timing_windows'] == len(info['schedule']), 'Nine-group schedule changed')
    seal = read(root/'provenance/preparation.json')
    require(all(seal[k] == info[k] for k in ('protocol', 'run_id', 'source_only', 'executable')) and
            info['executable'] == (not info['source_only']), 'Rehearsal flags changed')
    plans = read(root/'scripts/selection-plan.json'); cfg = read(root/'scripts/config.json')
    require(type(seal.get('rounds', 2)) is int and seal.get('rounds', 2) == rounds and
            type(cfg['repetitions']) is int and cfg['repetitions'] == rounds, 'Frozen round count changed')
    hashes = {p: {r: v['entry_sha256'] for r, v in data['roles'].items()} for p, data in plans['variants'].items()}
    require(sha(root/'scripts/selection-plan.json') == seal['plans_sha256'] and cfg['protocol'] == PROTOCOL and
            cfg['policies'] == list(POLICIES) and cfg['role_plan_sha256'] == hashes, 'Worker policy/plan identity changed')
    m = read(root/'provenance/build-manifest.json')
    require(sha(root/'provenance/build-manifest.json') == info['build_manifest']['sha256'] and
            m['files']['plans.json'] == seal['plans_sha256'], 'Build/plan snapshot mismatch')
    children = {order: read(root/'provenance'/f'build-{order}-manifest.json') for order in builder.ORDERS}
    builder.validate_snapshots(m, plans, children, built=info['executable'], verify_evidence=not offline)
    if info['executable']:
        require(not plans['source_only'] and all(hardware_identity(p['hardware']) == hardware_identity(info['calibration_hardware'])
                for p in plans['parent']['roles'].values()), 'Real role calibration/device identity required')
    require(set(info['variants']) == set(POLICIES), 'Missing/extra nine-group policy')
    libraries = {}; checked_libraries = set()
    for p in POLICIES:
        v = info['variants'][p]; d = definition(p); order = d['order']
        require(all(v.get(k) == value for k, value in d.items()) and v['sampling'] is False, 'Wrong policy/build/sampling route')
        identity = read(root/'provenance'/f'build-{order}-identity.json')
        require(identity['sampling'] is False and identity['source_only'] == children[order]['source_only'], 'Child build identity mismatch')
        pair = (v['root'], v['libraries'])
        if order in libraries:
            require(libraries[order] == pair, 'Same-order scopes must share libraries (B/C included)')
        else:
            libraries[order] = pair
        expected_root = Path(info['build_manifest']['path']).parent/order/'scope-candidate'
        require(Path(v['root']) == expected_root, 'Wrong policy library/order root')
        if info['executable']:
            require(set(v['libraries']) == {'libflux_cuda.so', 'libflux_cuda_ths_op.so'}, 'Missing runtime libraries')
            if not offline:
                for lib in v['libraries'].values():
                    identity = (lib['path'], lib['sha256'])
                    if identity not in checked_libraries:
                        require(sha(lib['path']) == lib['sha256'], 'Runtime library changed')
                        checked_libraries.add(identity)
    if not offline:
        require(sha(info['build_manifest']['path']) == info['build_manifest']['sha256'], 'Build manifest changed')
        verify_files('/', info['build_files'])
    return info


def window(root, folder, policy, mode, block=-1, index=-1):
    return scope_report.window(root, folder, policy, mode, block, index, routing)


def profile_roles(events, policy):
    return scope_report.profile_roles(events, policy, routing)


def preflight(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline)
    require(info['executable'] and not info['source_only'], 'CPU rehearsal cannot certify GPU execution')
    start = read(root/'start-gpu-check.json')
    require(start['passed'] and start['run_id'] == info['run_id'], 'Missing startup check')
    files = [root/'start-gpu-check.json']; reference = None; profiles = {}
    plans = read(root/'scripts/selection-plan.json')
    for policy in POLICIES:
        for mode in ('smoke', 'profile'):
            folder = root/'results'/(mode+'-'+policy)
            rows = window(root, folder, policy, mode); key = fingerprints(rows)
            if reference is None:
                reference = key
            require(key == reference, 'Cross-policy initialization/RNG/tokens/software mismatch')
            files += [p for p in folder.rglob('*') if p.is_file()]
        for rank in range(8):
            with gzip.open(root/'results'/('profile-'+policy)/f'rank{rank}-profile.json.gz', 'rt') as f:
                profiles[f'{policy}/rank{rank}'] = profile_roles(json.load(f)['traceEvents'], policy)
        for role, record in routing.role_plans(plans, policy).items():
            path = root/'results'/f'mapping-{policy}-{role}.csv'
            with path.open() as f:
                rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
            require(len(rows) == 4096 and {r['rank'] for r in rows} == set(range(8)), 'Incomplete GPU mapping')
            for rank in range(8):
                rr = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
                require([r['index'] for r in rr] == list(range(512)) and
                        [r['m']*32+r['n'] for r in rr] == record['entry']['coords'][rank] and
                        all(r['destination'] == r['m']//2 and r['sm_occupancy'] > 0 for r in rr), 'Wrong policy/role mapping')
            files.append(path)
    gate = dict(protocol=PROTOCOL, completed=True, submission_sha256=sha(root/'submission.json'),
                fingerprints=reference, role_profiles=profiles, files={str(p.relative_to(root)): sha(p) for p in files})
    write(root/'preflight.json', gate)
    return gate


def summarize(values, blocks, policies=POLICIES, pairs=COMPARISONS):
    require(set(values) == set(policies) and all(len(v) == len(blocks) for v in values.values()), 'Incomplete policy pairing')
    require(all(math.isfinite(x) and x > 0 for v in values.values() for x in v), 'Invalid timings')
    if not blocks:
        return dict(status='无可用配对块', blocks=0, comparisons={}, speedups={})
    comparisons = {}; speedups = {}
    for a, b in pairs:
        logs = [math.log(x/y) for x, y in zip(values[a], values[b])]
        stat = paired(logs) if len(blocks) > 1 else dict(percent=100*(1-math.exp(logs[0])), ci95=None, blocks=1)
        key = a+'/'+b
        comparisons[key] = dict(stat, log_ratios=logs)
        ratios = [y/x for x, y in zip(values[a], values[b])]; best = max(range(len(ratios)), key=ratios.__getitem__)
        speedups[key] = dict(candidate=a, reference=b, median=statistics.median(ratios), ratios=ratios,
                            best=dict(ratio=ratios[best], **blocks[best], reference_ms=values[b][best], candidate_ms=values[a][best]))
    return dict(status='ok', blocks=len(blocks), block_ids=blocks, ms_per_step=values, comparisons=comparisons, speedups=speedups,
                median_ms={k: statistics.median(v) for k, v in values.items()},
                median_tokens_per_second={k: statistics.median(8192000/x for x in v) for k, v in values.items()})


def sensitivity(values, blocks, policies=POLICIES, pairs=COMPARISONS):
    medians = {k: statistics.median(v) for k, v in values.items()}; keep = []; excluded = []
    for i, block in enumerate(blocks):
        triggers = {k: dict(ms=values[k][i], threshold_ms=1.1*medians[k]) for k in policies if values[k][i] > 1.1*medians[k]}
        if triggers:
            excluded.append(dict(block, triggers=triggers))
        else:
            keep.append(i)
    return dict(original_median_ms=medians, excluded=excluded, remaining_blocks=len(keep),
                summary=summarize({k: [v[i] for i in keep] for k, v in values.items()}, [blocks[i] for i in keep], policies, pairs))


def markdown(result):
    lines = ['# H100 GPT6.7B TP8 九组消融', '',
             f'本次{result["round_count"]}轮，每轮{result["blocks_per_round"]}个完整配对块，'
             f'共{result["timing_windows"]}个正式窗口。',
             '所有ms/step均为完整32层模型、4微批的optimizer step。九组同块，全样本为主。',
             'A=交替+全远端量化；B=交替+到达优先+全远端量化；C=交替+到达优先+选择性量化。',
             '本地贡献及未选中模块保持BF16；没有同期native_taco或固定融合v2基线。',
             '95% CI按完整九组配对块bootstrap 10000次、seed20261009，逐项区间未校正多重比较；无4%门槛。', '']
    if result['round_count'] == 1:
        lines += ['单轮保留原第一轮的循环移位调度，没有第二轮反转验证；combined与第1轮是同一批数据，不增加样本数。', '']
    for label, rounds in [('全样本', result['rounds']), ('慢窗口敏感性附表', {k: v['summary'] for k, v in result['sensitivity'].items()})]:
        lines += [f'## {label}', '', '| 轮次 | 策略 | 中位ms/step | tokens/s |', '| --- | --- | ---: | ---: |']
        for rep, s in rounds.items():
            if not s['blocks']:
                lines.append(f'| {rep} | 无可用配对块 | — | — |')
            for k, ms in s.get('median_ms', {}).items():
                lines.append(f'| {rep} | {k} | {ms:.3f} | {s["median_tokens_per_second"][k]:.2f} |')
        for name, pairs in [('同范围策略比较', PRIMARY), ('同策略范围比较', RANGE_COMPARISONS)]:
            lines += ['', f'### {name}', '', '| 轮次 | X / 参考Y | 耗时下降% | 95% CI | 中位加速比 | 最好单块加速比 | 最好轮/block | Y ms / X ms |',
                      '| --- | --- | ---: | --- | ---: | ---: | --- | --- |']
            for rep, s in rounds.items():
                for a, b in pairs:
                    key = a+'/'+b
                    if key not in s['comparisons']:
                        continue
                    v = s['comparisons'][key]; speed = s['speedups'][key]; best = speed['best']; ci = v['ci95']
                    interval = f'[{ci[0]:.3f},{ci[1]:.3f}]' if ci else '不足两块'
                    lines.append(f'| {rep} | {key} | {v["percent"]:.3f} | {interval} | {speed["median"]:.6f} | '
                                 f'{best["ratio"]:.6f} | {best["round"]}/{best["block"]} | {best["reference_ms"]:.3f} / {best["candidate_ms"]:.3f} |')
    lines += ['', '最好值仅为单块观测；负值和跨零区间保留。范围差异不等于局部算子耗时，也不能直接相加。']
    for rep, s in result['sensitivity'].items():
        lines.append(f'筛选轮次{rep}：剩余{s["remaining_blocks"]}块；触发与排除：`{s["excluded"]}`。')
    return '\n'.join(lines)+'\n'


def report(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline); gate = read(root/'preflight.json')
    require(info['executable'] and not info['source_only'] and gate['completed'] and gate['protocol'] == PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json'), 'Completed GPU preflight required')
    verify_files(root, gate['files'])
    rounds = info.get('rounds', 2)
    expected = {job['directory'] for job in info['schedule']}
    observed = {str(path.relative_to(root/'results'))
                for path in (root/'results').glob('round*/window-*') if path.is_dir()}
    require(observed == expected, 'Missing or extra timing window directories')
    result = dict(protocol=PROTOCOL, run_id=info['run_id'], completed=False, offline=offline, all_samples=[], rounds={}, sensitivity={},
                  round_count=rounds, blocks_per_round=info['blocks_per_round'],
                  acceptance_threshold=None, sample_policy='all samples primary', correctness_passed=True,
                  bootstrap=dict(resamples=10000, seed=20261009, unit='complete nine-policy block', multiple_comparison_correction=None),
                  primary_comparisons=PRIMARY, scope_comparisons=RANGE_COMPARISONS,
                  config=info['config'], candidate_source_commit=info['candidate_source_commit'],
                  measurement_identity=measurement_identity(info, gate),
                  role_plan_sha256=read(root/'scripts/config.json')['role_plan_sha256'])
    combined = {k: [] for k in POLICIES}; combined_blocks = []; files = {}
    for rep in range(1, rounds + 1):
        values = {k: [] for k in POLICIES}; blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == 9 and {j['policy'] for j in jobs} == set(POLICIES), 'Incomplete nine-policy block')
            for job in jobs:
                folder = root/'results'/job['directory']
                rows = window(root, folder, job['policy'], 'timing', job['block'], job['index'])
                require(fingerprints(rows) == [tuple(x) for x in gate['fingerprints']], 'Timing initialization/RNG/tokens/software changed')
                elapsed = [r['elapsed_seconds'] for r in rows]; ms = max(elapsed)*1000/20
                values[job['policy']].append(ms)
                result['all_samples'].append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms, tokens_per_second=8192000/ms))
                files.update({str(p.relative_to(root)): sha(p) for p in folder.rglob('*') if p.is_file()})
        result['rounds'][str(rep)] = summarize(values, blocks); result['sensitivity'][str(rep)] = sensitivity(values, blocks)
        for k in POLICIES:
            combined[k] += values[k]
        combined_blocks += blocks
    result['rounds']['combined'] = summarize(combined, combined_blocks)
    require(len(result['all_samples']) == info['timing_windows'], 'Missing timing windows')
    result.update(completed=True, timing_windows=info['timing_windows'], timing_rank_records=8*info['timing_windows'],
                  preflight_rank_records=144, profile_traces=72, candidate_shape_mappings=12,
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=files))
    write(root/'report.json', result); (root/'report.md').write_text(markdown(result))
    return result
