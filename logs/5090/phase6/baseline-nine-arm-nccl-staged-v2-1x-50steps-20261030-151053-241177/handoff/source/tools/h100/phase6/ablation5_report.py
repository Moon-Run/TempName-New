"""Strict budget/codec identity, fresh preflight and complete paired reports."""
import csv
import gzip
import json
from pathlib import Path
import statistics

import ablation3_report
import ablation4_runtime as communication
import ablation5_plan as planner
import ablation5_protocol as routing
from ablation5_prepare import candidate_snapshot, role_hashes
from ablation_report import measurement_identity
from common import read, require, sha, verify_files, write
from report import fingerprints
from scope_plan import model_config, hardware_identity
import scope_report
import ablation_timing

PROTOCOL, POLICIES = routing.PROTOCOL, routing.POLICIES


def verify_inputs(root, offline=False):
    root = Path(root); info = read(root/'submission.json')
    require(info['protocol'] == PROTOCOL and info['config'] == model_config() and
            info['budget_denominators'] == list(routing.DENOMINATORS), 'Wrong ablation5 model/budgets')
    verify_files(root, info['files'])
    expected = routing.schedule(info['blocks_per_round'], info['schedule_seed'], info['rounds'])
    require(info['schedule'] == read(root/'schedule.json') == expected and info['timing_windows'] == len(expected), 'Budget schedule changed')
    seal = read(root/'provenance/preparation.json'); cfg = read(root/'scripts/config.json')
    require(all(seal[k] == info[k] for k in ('protocol', 'run_id', 'source_only', 'executable', 'rounds', 'communication')) and
            info['executable'] == (not info['source_only']), 'Execution/communication identity changed')
    plan = planner.validate(read(root/'scripts/selection-plan.json'), verify_evidence=not offline)
    require(plan['source_only'] == info['source_only'] and sha(root/'scripts/selection-plan.json') == seal['plans_sha256'], 'Plan identity changed')
    require(cfg['protocol'] == PROTOCOL and cfg['policies'] == list(POLICIES) and cfg['repetitions'] == info['rounds'] and
            cfg['communication'] == communication.validate(info['communication']) and cfg['role_plan_sha256'] == role_hashes(plan) and
            cfg['budget_fractions'] == {p: routing.definition(p)['budget_fraction'] for p in POLICIES}, 'Budget worker changed')
    ablation_timing.validate(root, info)
    require(not cfg['graph_forward'] and
            cfg['double_buffered'] is True and cfg['ring_reduction'] is True, 'C execution path changed')
    model = model_config()
    require(cfg['common'] == {k: model[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')} and
            cfg['cases'] == [dict(name='h100-phase6', **{k: model[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
            'Worker model/full shapes changed')
    manifest = read(root/'provenance/candidate-manifest.json')
    require(sha(root/'provenance/candidate-manifest.json') == info['build_manifest']['sha256'], 'C manifest changed')
    candidate_snapshot(manifest, plan['parent'], info['source_only'], built=info['executable'])
    if info['executable']:
        require(all(hardware_identity(p['hardware']) == hardware_identity(info['calibration_hardware'])
                    for p in plan['parent']['roles'].values()), 'Budget calibration hardware/order changed')
    require(set(info['variants']) == set(POLICIES), 'Missing/extra budget policy')
    expected_root = Path(info['build_manifest']['path']).parent/'scope-candidate'
    common = None
    for policy in POLICIES:
        variant = info['variants'][policy]
        require(all(variant.get(k) == v for k, v in routing.definition(policy).items()) and not variant['sampling'] and
                Path(variant['root']) == expected_root, 'Budget implementation/library/order changed')
        identity = (variant['root'], variant['libraries'])
        if common is None:
            common = identity
            if info['executable']:
                require(set(variant['libraries']) == {'libflux_cuda.so', 'libflux_cuda_ths_op.so'}, 'Missing C libraries')
                if not offline:
                    for library in variant['libraries'].values():
                        require(sha(library['path']) == library['sha256'], 'C runtime library changed')
        require(identity == common, 'All five budgets must use the exact same library')
    if not offline:
        verify_files('/', info['build_files'])
        require(sha(info['build_manifest']['path']) == info['build_manifest']['sha256'], 'C build manifest changed')
    return info


def window(root, folder, policy, mode, block=-1, index=-1):
    rows = scope_report.window(root, folder, policy, mode, block, index, routing)
    info = read(Path(root)/'submission.json'); command = read(Path(folder)/'command.json')
    require(command['communication'] == info['communication'] and command['scope'] == routing.definition(policy) and
            command['policy'] == policy and command['mode'] == mode and command['block'] == block and command['index'] == index,
            'Budget command identity changed')
    require(all(r['communication'] == info['communication'] and r['losses'] == rows[0]['losses'] for r in rows),
            'Communication or replicated TP loss differs across ranks')
    return rows


def profile_roles(events, policy):
    result = scope_report.profile_roles(events, policy, routing)
    for role in routing.SHAPES:
        decoders = [n for n in result[role]['kernels'] if 'taco_decode_ring_kernel' in n]
        require(decoders and all('taco_decode_ring_kernel<' in n for n in decoders), 'Candidate warp decoder required')
    return result


def preflight(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline)
    require(info['executable'] and not info['source_only'], 'CPU rehearsal cannot certify GPU preflight')
    start = read(root/'start-gpu-check.json')
    require(start['passed'] and start['run_id'] == info['run_id'], 'Missing startup check')
    plan = read(root/'scripts/selection-plan.json'); reference = None; profiles = {}; files = [root/'start-gpu-check.json']
    for policy in POLICIES:
        for mode in ('smoke', 'profile'):
            folder = root/'results'/f'{mode}-{policy}'
            key = fingerprints(window(root, folder, policy, mode))
            if reference is None: reference = key
            require(key == reference, 'Budget initialization/RNG/token/software differs')
            files += [p for p in folder.rglob('*') if p.is_file()]
        for rank in range(8):
            with gzip.open(root/'results'/f'profile-{policy}'/f'rank{rank}-profile.json.gz', 'rt') as f:
                profiles[f'{policy}/rank{rank}'] = profile_roles(json.load(f)['traceEvents'], policy)
        for role, record in routing.role_plans(plan, policy).items():
            path = root/'results'/f'mapping-{policy}-{role}.csv'
            with path.open() as f: rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
            require(len(rows) == 4096 and {r['rank'] for r in rows} == set(range(8)), 'Incomplete mapping evidence')
            for rank in range(8):
                rr = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
                require([r['index'] for r in rr] == list(range(512)) and
                        [r['m']*32+r['n'] for r in rr] == record['entry']['coords'][rank] and
                        all(r['destination'] == r['m']//2 and r['sm_occupancy'] > 0 for r in rr), 'Budget changed arrival mapping')
            files.append(path)
    gate = dict(protocol=PROTOCOL, completed=True, submission_sha256=sha(root/'submission.json'),
                communication=info['communication'], fingerprints=reference, role_profiles=profiles,
                files={str(p.relative_to(root)): sha(p) for p in files})
    write(root/'preflight.json', gate)
    return gate


def summarize(values, blocks, samples):
    result = ablation3_report.summarize(values, blocks, POLICIES, routing.COMPARISONS)
    if blocks:
        result['mean_ms'] = {p: statistics.mean(v) for p, v in values.items()}
        result['tokens_per_second'] = {p: 8192000/v for p, v in result['mean_ms'].items()}
        result['mean_loss'] = {p: statistics.mean(s['mean_loss'] for s in samples if s['policy'] == p) for p in POLICIES}
        result['mean_final_loss'] = {p: statistics.mean(s['final_loss'] for s in samples if s['policy'] == p) for p in POLICIES}
    return result


def budget_metadata(plans):
    return {p: {r: dict(selected_per_source=v['entry']['selected_per_source'],
                         selected_total=sum(v['entry']['selected_per_source']), remote_total=3584,
                         selected_remote_fraction=v['selected_remote_fraction'],
                         wire_bytes_per_source=v['wire_bytes_per_source'], mask_sha256=v['mask_sha256'])
                for r, v in routing.role_plans(plans, p).items()} for p in POLICIES}


def markdown(result):
    labels = routing.LABELS
    lines = ['# H100 消融实验5：选择性量化预算上限', '',
        f'协议 `{PROTOCOL}`；{result["round_count"]}轮、每轮{result["blocks_per_round"]}个完整五组块，'
        f'{result["timing_windows"]}个正式窗口。MLP＋Attention均启用原C候选。',
        f'每窗口预热{result["execution_timing"]["warmup_steps"]}步，正式计时{result["execution_timing"]["timed_steps"]}步；'
        f'合计{result["formal_optimizer_steps"]}个正式optimizer step（不重复计rank）。',
        '五组共用同一运行库、到达表、H128/E4M3 codec、双工作区与BF16/STE反向，只改正分远端贡献的mask预算。',
        '预算为上限，不补齐零分贡献；更大预算可能产生相同mask，不能按名义比例解释为实际流量变化。', '',
        '| 预算 | Attention实际量化/3584 | MLP实际量化/3584 | Attention远端字节/投影（八来源合计） | MLP远端字节/投影 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for p in POLICIES:
        a, m = (result['budgets'][p][r] for r in ('attn', 'mlp'))
        lines.append(f'| {labels[p]} | {a["selected_total"]} ({100*a["selected_remote_fraction"]:.3f}%) | '
                     f'{m["selected_total"]} ({100*m["selected_remote_fraction"]:.3f}%) | '
                     f'{sum(a["wire_bytes_per_source"])} | {sum(m["wire_bytes_per_source"])} |')
    for rep, s in result['rounds'].items():
        lines += ['', f'## {rep}', '', '| 指标 | '+' | '.join(labels[p] for p in POLICIES)+' |',
                  '| --- | '+' | '.join(['---:']*5)+' |']
        for field, label, digits in (('mean_ms', '平均完整step耗时（ms）', 3), ('median_ms', '中位耗时（ms）', 3),
                                     ('mean_loss', '平均训练loss', 6), ('mean_final_loss', '最终训练loss的平均', 6)):
            lines.append('| '+label+' | '+' | '.join(f'{s[field][p]:.{digits}f}' for p in POLICIES)+' |')
        lines += ['', '| X / 参考Y | 配对耗时下降% [95% CI] |', '| --- | ---: |']
        for key, v in s['comparisons'].items():
            a, b = key.split('/'); ci = v['ci95']
            interval = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '不足两块'
            lines.append(f'| {labels[a]} / {labels[b]} | {v["percent"]:+.3f} {interval} |')
    lines += ['', '全样本为主；配对区间按完整五组块bootstrap10000次、seed20261009，未做多重比较校正，无4%门槛。',
              f'loss只计rank0的{result["execution_timing"]["timed_steps"]}个正式step，末步loss按各窗口losses[-1]平均；不含预热，不表示长期收敛。',
              '每轮按同组中位数×1.10触发整块排除，敏感性配对结果保存在report.json。']
    for rep, s in result['sensitivity'].items():
        lines.append(f'轮次{rep}：保留{s["remaining_blocks"]}块，排除：`{s["excluded"]}`。')
    return '\n'.join(lines)+'\n'


def report(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline); gate = read(root/'preflight.json')
    timing = ablation_timing.validate(root, info)
    require(info['executable'] and not info['source_only'] and gate['completed'] and gate['protocol'] == PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json') and gate['communication'] == info['communication'],
            'Fresh complete preflight required')
    verify_files(root, gate['files'])
    observed = {str(p.relative_to(root/'results')) for p in (root/'results').glob('round*/window-*') if p.is_dir()}
    require(observed == {j['directory'] for j in info['schedule']}, 'Missing or extra timing windows')
    plans = read(root/'scripts/selection-plan.json')
    result = dict(protocol=PROTOCOL, run_id=info['run_id'], completed=False, offline=offline, all_samples=[], rounds={}, sensitivity={},
        round_count=info['rounds'], blocks_per_round=info['blocks_per_round'], communication=info['communication'],
        correctness_passed=True, acceptance_threshold=None, config=info['config'], budgets=budget_metadata(plans),
        execution_timing=timing, formal_optimizer_steps=info['timing_windows']*timing['timed_steps'],
        budget_denominators=list(routing.DENOMINATORS), candidate_source_commit=info['candidate_source_commit'],
        measurement_identity=measurement_identity(info, gate), role_plan_sha256=role_hashes(plans),
        primary_comparisons=routing.PRIMARY, comparisons=routing.COMPARISONS,
        bootstrap=dict(resamples=10000, seed=20261009, unit='complete five-policy block', multiple_comparison_correction=None))
    result['measurement_identity']['communication'] = info['communication']
    combined = {p: [] for p in POLICIES}; combined_blocks = []; files = {}
    for rep in range(1, info['rounds']+1):
        values = {p: [] for p in POLICIES}; blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]
        samples = []
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == 5 and {j['policy'] for j in jobs} == set(POLICIES), 'Incomplete budget block')
            for job in jobs:
                folder = root/'results'/job['directory']
                rows = window(root, folder, job['policy'], 'timing', job['block'], job['index'])
                require(fingerprints(rows) == [tuple(x) for x in gate['fingerprints']], 'Timing identity differs from preflight')
                elapsed = [r['elapsed_seconds'] for r in rows]; ms = max(elapsed)*1000/timing['timed_steps']
                values[job['policy']].append(ms)
                samples.append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms, tokens_per_second=8192000/ms,
                                    mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1]))
                files.update({str(p.relative_to(root)): sha(p) for p in folder.rglob('*') if p.is_file()})
        result['rounds'][str(rep)] = summarize(values, blocks, samples)
        result['sensitivity'][str(rep)] = ablation3_report.sensitivity(values, blocks, POLICIES, routing.COMPARISONS)
        for p in POLICIES: combined[p] += values[p]
        combined_blocks += blocks; result['all_samples'] += samples
    result['rounds']['combined'] = summarize(combined, combined_blocks, result['all_samples'])
    require(len(result['all_samples']) == info['timing_windows'], 'Missing timing windows')
    result.update(completed=True, timing_windows=info['timing_windows'], timing_rank_records=8*info['timing_windows'],
                  preflight_rank_records=80, profile_traces=40, candidate_shape_mappings=10,
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=files))
    write(root/'report.json', result); (root/'report.md').write_text(markdown(result))
    return result
