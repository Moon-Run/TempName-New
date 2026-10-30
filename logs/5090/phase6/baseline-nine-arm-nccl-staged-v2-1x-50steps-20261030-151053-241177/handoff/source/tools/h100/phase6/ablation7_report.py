"""Strict C/A/improved-B evidence and complete three-policy paired reports."""
import csv
import gzip
import json
from pathlib import Path
import statistics

import ablation3_report
import ablation4_runtime as communication
import ablation7_build as builder
import ablation7_protocol as routing
from ablation_report import measurement_identity
from common import read, require, sha, verify_files, write
from report import fingerprints
from scope_plan import model_config, hardware_identity
import scope_report

PROTOCOL, POLICIES = routing.PROTOCOL, routing.POLICIES


def role_hashes(plans):
    return {p: {role: record['entry_sha256'] for role, record in routing.role_plans(plans, p).items()}
            for p in POLICIES}


def validate_timing(info, cfg):
    require(info['execution_timing'] == cfg['execution_timing'] == routing.TIMING and
            all(type(cfg[k]) is int and cfg[k] == v for k, v in routing.TIMING.items()),
            'Ablation7 requires 10 warmup / 50 timed steps')
    return dict(routing.TIMING)


def budget_metadata(plans):
    """Report the physical masks, including budgets that select fewer than 28."""
    result = {}
    for policy in POLICIES:
        selective = not routing.is_fused_v2(policy)
        roles = {}
        for role, record in routing.role_plans(plans, policy).items():
            mask = record['entry']['mask']
            require(len(mask) == 8 and all(len(row) == 512 for row in mask), 'Wrong full-shape mask')
            require(all(value in (0, 1) for row in mask for value in row), 'Mask must be binary')
            require(all(not mask[rank][tile] for rank in range(8)
                        for tile in range(64*rank, 64*(rank+1))), 'Local contributions must remain BF16')
            counts = list(map(sum, mask))
            require(all(q <= 28 for q in counts) if selective else counts == [448]*8,
                    'C requires a 1/16 budget; A/B require all 448 remote tiles')
            roles[role] = dict(selected_per_source=counts, selected_remote_fraction=sum(counts)/3584,
                               mask_sha256=record['mask_sha256'], entry_sha256=record['entry_sha256'],
                               wire_bytes_per_source=[14680064-q*128*(256-136) for q in counts])
        result[policy] = dict(budget_fraction=1/16 if selective else 1,
                              max_remote_tiles_per_source=28 if selective else 448,
                              local_quantization=False, roles=roles)
    return result


def arrival_metadata(plans):
    """Keep identity fallback visible instead of claiming a changed GPU order."""
    base = routing.role_plans(plans, routing.A)
    improved = routing.role_plans(plans, routing.B)
    return {role: dict(method=plans['orders'][role]['entry']['arrival_order']['method'],
                       training_passes=plans['orders'][role]['training_passes'],
                       heldout_passes=plans['orders'][role]['heldout_passes'],
                       changed_per_rank=[sum(a != b for a, b in zip(base[role]['entry']['coords'][rank],
                                                                   improved[role]['entry']['coords'][rank]))
                                         for rank in range(8)],
                       identical_to_A=improved[role]['entry']['coords'] == base[role]['entry']['coords'],
                       fitted_entry_sha256=plans['orders'][role]['entry_sha256'])
            for role in routing.SHAPES}


def verify_inputs(root, offline=False):
    root = Path(root); info = read(root/'submission.json')
    require(info['protocol'] == PROTOCOL and info['config'] == model_config(), 'Wrong ablation7 protocol/model')
    verify_files(root, info['files'])
    expected = routing.schedule(info['blocks_per_round'], info['schedule_seed'], info['rounds'])
    require(info['schedule'] == read(root/'schedule.json') == expected and info['timing_windows'] == len(expected),
            'Three-policy balanced schedule changed')
    seal = read(root/'provenance/preparation.json'); cfg = read(root/'scripts/config.json')
    require(all(seal[k] == info[k] for k in ('protocol', 'run_id', 'source_only', 'executable', 'rounds',
                                          'communication', 'execution_timing')) and
            info['executable'] == (not info['source_only']), 'Preparation/execution identity changed')
    validate_timing(info, cfg)
    require(info['budget_fractions'] == cfg['budget_fractions'] ==
            {p: routing.definition(p)['budget_fraction'] for p in POLICIES}, 'Quantization budgets changed')
    require(cfg['protocol'] == PROTOCOL and cfg['policies'] == list(POLICIES) and cfg['repetitions'] == info['rounds'] and
            cfg['communication'] == communication.validate(info['communication']) and
            cfg['graph_forward'] is False and cfg['double_buffered'] is True and cfg['ring_reduction'] is True,
            'Worker execution path changed')
    model = model_config()
    require(cfg['common'] == {k: model[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')} and
            cfg['cases'] == [dict(name='h100-phase6', **{k: model[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
            'Worker model/full shapes changed')
    plans = read(root/'scripts/selection-plan.json'); manifest = read(root/'provenance/build-manifest.json')
    candidate = read(root/'provenance/parent-candidate-manifest.json'); fused = read(root/'provenance/parent-fused-manifest.json')
    improved = read(root/'provenance/improved-manifest.json')
    require(sha(root/'scripts/selection-plan.json') == seal['plans_sha256'] == manifest['files']['plans.json'] and
            cfg['role_plan_sha256'] == role_hashes(plans), 'Role plans/order hashes changed')
    require(sha(root/'provenance/build-manifest.json') == info['build_manifest']['sha256'], 'Build snapshot changed')
    for name, relative in (('parent-ablation4-manifest.json', 'parent-ablation4-manifest.json'),
                           ('parent-candidate-manifest.json', 'parent-candidate-manifest.json'),
                           ('parent-fused-manifest.json', 'parent-fused-manifest.json'),
                           ('parent-plans.json', 'parent-plans.json'), ('improved-manifest.json', 'improved/manifest.json')):
        require(sha(root/'provenance'/name) == manifest['files'][relative], 'Frozen build snapshot changed: '+name)
    builder.validate_snapshots(manifest, plans, candidate, fused, improved, built=info['executable'], verify_evidence=not offline)
    require(manifest['source_only'] == plans['source_only'] == info['source_only'], 'Build rehearsal identity changed')
    budget_metadata(plans)
    if info['executable']:
        require(all(hardware_identity(r['hardware']) == hardware_identity(info['calibration_hardware'])
                    for r in plans['parent']['roles'].values()), 'Calibration hardware/order changed')
    require(set(info['variants']) == set(POLICIES), 'Missing/extra ablation7 policies')
    build_root = Path(info['build_manifest']['path']).parent
    for policy in POLICIES:
        variant = info['variants'][policy]
        require(all(variant.get(k) == value for k, value in routing.definition(policy).items()) and
                variant['sampling'] is False and Path(variant['root']) == builder.runtime_root(build_root, policy, record=manifest),
                'Wrong policy implementation/order/build route')
        if info['executable']:
            require(set(variant['libraries']) == {'libflux_cuda.so', 'libflux_cuda_ths_op.so'}, 'Missing runtime libraries')
            if not offline:
                for library in variant['libraries'].values():
                    require(sha(library['path']) == library['sha256'], 'Runtime library changed')
    if not offline:
        verify_files('/', info['build_files'])
        require(sha(info['build_manifest']['path']) == info['build_manifest']['sha256'], 'Build manifest changed')
    return info


def window(root, folder, policy, mode, block=-1, index=-1):
    root, folder = Path(root), Path(folder)
    info = read(root/'submission.json'); validate_timing(info, read(root/'scripts/config.json'))
    rows = scope_report.window(root, folder, policy, mode, block, index, routing)
    command = read(folder/'command.json')
    require(command['communication'] == info['communication'] and command['scope'] == routing.definition(policy) and
            all(command[k] == v for k, v in dict(policy=policy, mode=mode, block=block, index=index).items()),
            'Command communication/policy identity changed')
    require(all(r['communication'] == info['communication'] and r['losses'] == rows[0]['losses'] and
                r['execution_timing'] == routing.TIMING for r in rows), 'TP loss/communication/timing mismatch')
    return rows


def profile_roles(events, policy):
    return scope_report.profile_roles(events, policy, routing)


def verify_mapping(rows, policy, role, plans):
    require(len(rows) == 4096 and {r['rank'] for r in rows} == set(range(8)), 'Incomplete GPU mapping')
    record = routing.role_plans(plans, policy)[role]
    for rank in range(8):
        mapped = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
        require([r['index'] for r in mapped] == list(range(512)) and
                sorted(r['m']*32+r['n'] for r in mapped) == list(range(512)) and
                [r['m']*32+r['n'] for r in mapped] == record['entry']['coords'][rank] and
                all(r['destination'] == r['m']//2 and r['sm_occupancy'] > 0 for r in mapped),
                'Wrong policy/role GPU mapping')


def preflight(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline)
    require(info['executable'] and not info['source_only'], 'CPU rehearsal cannot certify GPU execution')
    start = read(root/'start-gpu-check.json')
    require(start['passed'] and start['run_id'] == info['run_id'], 'Missing startup check')
    files = [root/'start-gpu-check.json']; reference = None; profiles = {}
    plans = read(root/'scripts/selection-plan.json')
    for policy in POLICIES:
        for mode in ('smoke', 'profile'):
            folder = root/'results'/f'{mode}-{policy}'; rows = window(root, folder, policy, mode)
            key = fingerprints(rows)
            if reference is None: reference = key
            require(key == reference, 'Cross-policy initialization/RNG/tokens/software mismatch')
            files += [f for f in folder.rglob('*') if f.is_file()]
        for rank in range(8):
            with gzip.open(root/'results'/f'profile-{policy}'/f'rank{rank}-profile.json.gz', 'rt') as f:
                profiles[f'{policy}/rank{rank}'] = profile_roles(json.load(f)['traceEvents'], policy)
        for role in routing.SHAPES:
            path = root/'results'/f'mapping-{policy}-{role}.csv'
            with path.open() as f:
                verify_mapping([{k: int(v) for k, v in row.items()} for row in csv.DictReader(f)], policy, role, plans)
            files.append(path)
    gate = dict(protocol=PROTOCOL, completed=True, submission_sha256=sha(root/'submission.json'),
                communication=info['communication'], fingerprints=reference, role_profiles=profiles,
                execution_timing=dict(routing.TIMING), files={str(p.relative_to(root)): sha(p) for p in files})
    write(root/'preflight.json', gate)
    return gate


def summarize(values, blocks, samples):
    result = ablation3_report.summarize(values, blocks, POLICIES, routing.COMPARISONS)
    if blocks:
        result['mean_ms'] = {p: statistics.mean(v) for p, v in values.items()}
        result['tokens_per_second'] = {p: 8192000/ms for p, ms in result['mean_ms'].items()}
        result['mean_loss'] = {p: statistics.mean(s['mean_loss'] for s in samples if s['policy'] == p) for p in POLICIES}
        result['mean_final_loss'] = {p: statistics.mean(s['final_loss'] for s in samples if s['policy'] == p) for p in POLICIES}
    return result


def markdown(result):
    lines = ['# H100 消融7：C、A与改进到达调度B', '',
             f'每窗口预热10步、正式计时50步；{result["timing_windows"]}个正式窗口，'
             f'{result["formal_optimizer_steps"]}个正式optimizer step（不重复计rank）。',
             '完整32层模型、4微批、8192 token/step；每窗口取8 rank最大elapsed/50。',
             'C保留原到达表和candidate实现，以1/16预算重新选择远端mask；A/B使用同一固定融合v2全远端量化实现。',
             'B只替换为tail_residual改进到达调度；无稳健收益证据时允许回退到A的交替表。',
             'B/A是调度净收益的主比较；C/A和C/B同时包含量化覆盖率与实现差异。',
             '本地贡献和GEMM保持BF16；1/16为上限，每来源最多28/448个远端tile，实际数量如下。', '',
             '| C角色 | 每来源实际选中数（rank 0–7） | 实际远端比例 |', '| --- | --- | ---: |']
    for role, metadata in result['budgets'][POLICIES[0]]['roles'].items():
        lines.append(f'| {role} | {metadata["selected_per_source"]} | {100*metadata["selected_remote_fraction"]:.4f}% |')
    for role, metadata in result['arrival_schedule'].items():
        lines.append(f'B的{role}表相对A每rank改变槽位数：{metadata["changed_per_rank"]}；'
                     f'完全回退到A：{metadata["identical_to_A"]}。')
    for rep, summary in result['rounds'].items():
        lines += ['', f'## {rep}', '', '| 指标 | '+' | '.join(routing.LABELS[p] for p in POLICIES)+' |',
                  '| --- | '+' | '.join(['---:']*3)+' |']
        for field, label, fmt in (('mean_ms', '平均耗时（ms/step）', '.3f'), ('median_ms', '中位耗时（ms/step）', '.3f'),
                                  ('tokens_per_second', 'tokens/s（总量/总时间）', '.2f'),
                                  ('mean_loss', '平均训练loss', '.6f'), ('mean_final_loss', '最终训练loss的平均', '.6e')):
            lines.append('| '+label+' | '+' | '.join(format(summary[field][p], fmt) for p in POLICIES)+' |')
        lines += ['', '| X / 参考Y | 配对耗时下降% [95% CI] | 配对中位加速比 |', '| --- | ---: | ---: |']
        for key, value in summary['comparisons'].items():
            a, b = key.split('/'); ci = value['ci95']
            interval = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '不足两块'
            lines.append(f'| {routing.LABELS[a]} / {routing.LABELS[b]} | {value["percent"]:+.3f} {interval} | '
                         f'{summary["speedups"][key]["median"]:.6f} |')
    lines += ['', '主表保留全样本，负收益和跨零区间均保留；完整三组块bootstrap10000次、seed20261009，'
              '无多重比较校正，无4%门槛。',
              'loss只计rank0的50个正式step，末步loss跨窗口平均；固定合成微批重复训练不代表长期收敛。']
    if result['round_count'] == 1:
        lines.append('单轮没有第二轮反转验证；combined与第1轮为同一批样本。')
    for rep, sensitivity in result['sensitivity'].items():
        lines.append(f'慢窗口敏感性轮次{rep}：保留{sensitivity["remaining_blocks"]}块，排除：`{sensitivity["excluded"]}`。')
    return '\n'.join(lines)+'\n'


def report(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline)
    require(info['executable'] and not info['source_only'], 'CPU rehearsal cannot certify GPU execution')
    gate = read(root/'preflight.json')
    require(gate['completed'] and gate['protocol'] == PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json') and gate['communication'] == info['communication'] and
            gate['execution_timing'] == routing.TIMING, 'Fresh complete preflight required')
    verify_files(root, gate['files'])
    require({str(p.relative_to(root/'results')) for p in (root/'results').glob('round*/window-*') if p.is_dir()} ==
            {job['directory'] for job in info['schedule']}, 'Missing or extra timing windows')
    plans = read(root/'scripts/selection-plan.json')
    result = dict(protocol=PROTOCOL, run_id=info['run_id'], completed=False, correctness_passed=True, offline=offline,
                  round_count=info['rounds'], blocks_per_round=info['blocks_per_round'], config=info['config'],
                  execution_timing=dict(routing.TIMING), communication=info['communication'], all_samples=[], rounds={}, sensitivity={},
                  candidate_source_commit=info['candidate_source_commit'], role_plan_sha256=role_hashes(plans),
                  budgets=budget_metadata(plans), arrival_schedule=arrival_metadata(plans),
                  primary_comparisons=routing.PRIMARY, comparisons=routing.COMPARISONS,
                  acceptance_threshold=None, sample_policy='all samples primary', measurement_identity=measurement_identity(info, gate),
                  bootstrap=dict(resamples=10000, seed=20261009, unit='complete three-policy block', multiple_comparison_correction=None))
    result['measurement_identity']['communication'] = info['communication']
    combined = {p: [] for p in POLICIES}; all_blocks = []; files = {}
    for rep in range(1, info['rounds']+1):
        values = {p: [] for p in POLICIES}; blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]; samples = []
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == 3 and {j['policy'] for j in jobs} == set(POLICIES), 'Incomplete three-policy paired block')
            for job in jobs:
                folder = root/'results'/job['directory']; rows = window(root, folder, job['policy'], 'timing', job['block'], job['index'])
                require(fingerprints(rows) == [tuple(x) for x in gate['fingerprints']], 'Timing initialization/RNG/tokens/software changed')
                elapsed = [r['elapsed_seconds'] for r in rows]; ms = max(elapsed)*1000/routing.TIMING['timed_steps']
                values[job['policy']].append(ms)
                samples.append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms, tokens_per_second=8192000/ms,
                                    mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1]))
                files.update({str(p.relative_to(root)): sha(p) for p in folder.rglob('*') if p.is_file()})
        result['rounds'][str(rep)] = summarize(values, blocks, samples)
        result['sensitivity'][str(rep)] = ablation3_report.sensitivity(values, blocks, POLICIES, routing.COMPARISONS)
        for policy in POLICIES: combined[policy] += values[policy]
        all_blocks += blocks; result['all_samples'] += samples
    result['rounds']['combined'] = summarize(combined, all_blocks, result['all_samples'])
    require(len(result['all_samples']) == info['timing_windows'], 'Missing timing samples')
    result.update(completed=True, timing_windows=info['timing_windows'], formal_optimizer_steps=50*info['timing_windows'],
                  timing_rank_records=8*info['timing_windows'], preflight_rank_records=48, profile_traces=24, candidate_shape_mappings=6,
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=files))
    write(root/'report.json', result); (root/'report.md').write_text(markdown(result))
    return result
