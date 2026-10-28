"""Strict same-codec A/S scheduling evidence and independent full-step acceptance."""
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

import ablation3_report
import ablation4_runtime as communication
import arrival_op2_build as builder
import arrival_op2_protocol as routing
import arrival_op2_probe as probe
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
            'Arrival-op2 requires 10 warmup / 50 timed steps')
    return dict(routing.TIMING)


def budget_metadata(plans):
    """Both physical masks retain every remote contribution and exclude local tiles."""
    result = {}
    for policy in POLICIES:
        roles = {}
        for role, record in routing.role_plans(plans, policy).items():
            mask = record['entry']['mask']
            require(len(mask) == 8 and all(len(row) == 512 for row in mask), 'Wrong full-shape mask')
            require(all(value in (0, 1) for row in mask for value in row), 'Mask must be binary')
            require(all(not mask[rank][tile] for rank in range(8)
                        for tile in range(64*rank, 64*(rank+1))), 'Local contributions must remain BF16')
            counts = list(map(sum, mask))
            require(counts == [448]*8, 'A/S require all 448 remote tiles')
            roles[role] = dict(selected_per_source=counts, selected_remote_fraction=sum(counts)/3584,
                               mask_sha256=record['mask_sha256'], entry_sha256=record['entry_sha256'],
                               wire_bytes_per_source=[14680064-q*128*(256-136) for q in counts])
        result[policy] = dict(budget_fraction=1, max_remote_tiles_per_source=448,
                              local_quantization=False, roles=roles)
    return result


def arrival_metadata(plans):
    """Report actual producer order changes without calling locality an arrival fit."""
    base = routing.role_plans(plans, routing.A)
    candidate = routing.role_plans(plans, routing.S)
    return {role: dict(plans['schedule_diagnostics'][role],
                       identical_to_A=candidate[role]['entry']['coords'] == base[role]['entry']['coords'],
                       entry_sha256=candidate[role]['entry_sha256'])
            for role in routing.SHAPES}


def verify_inputs(root, offline=False):
    root = Path(root); info = read(root/'submission.json')
    require(info['protocol'] == PROTOCOL and info['config'] == model_config(), 'Wrong arrival_op2 protocol/model')
    verify_files(root, info['files'])
    expected = routing.schedule(info['blocks_per_round'], info['schedule_seed'], info['rounds'])
    require(info['schedule'] == read(root/'schedule.json') == expected and info['timing_windows'] == len(expected),
            'Two-policy balanced schedule changed')
    seal = read(root/'provenance/preparation.json'); cfg = read(root/'scripts/config.json')
    require(all(seal[k] == info[k] for k in ('protocol', 'run_id', 'source_only', 'executable', 'rounds',
                                          'communication', 'execution_timing')) and
            info['executable'] == (not info['source_only']), 'Preparation/execution identity changed')
    validate_timing(info, cfg)
    require(info['budget_fractions'] == cfg['budget_fractions'] ==
            {p: routing.definition(p)['budget_fraction'] for p in POLICIES}, 'Quantization budgets changed')
    require(cfg['protocol'] == PROTOCOL and cfg['policies'] == list(POLICIES) and cfg['repetitions'] == info['rounds'] and
            cfg['communication'] == communication.validate(info['communication']) and
            cfg['graph_forward'] is False and cfg['double_buffered'] is False and cfg['ring_reduction'] is True,
            'Worker execution path changed')
    model = model_config()
    require(cfg['common'] == {k: model[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')} and
            cfg['cases'] == [dict(name='h100-phase6', **{k: model[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
            'Worker model/full shapes changed')
    plans = read(root/'scripts/selection-plan.json'); manifest = read(root/'provenance/build-manifest.json')
    options = builder.planner.frozen_options(plans)
    require(all(builder.planner.frozen_options(record) == options for record in (info, cfg, seal)),
            'Submission/worker/preparation schedule changed')
    candidate = read(root/'provenance/parent-candidate-manifest.json'); fused = read(root/'provenance/parent-fused-manifest.json')
    improved = read(root/'provenance/improved-manifest.json')
    template = read(root/'provenance/build-plans.json')
    require(sha(root/'scripts/selection-plan.json') == seal['plans_sha256'] and
            sha(root/'provenance/build-plans.json') == seal['build_plans_sha256'] == manifest['files']['plans.json'] and
            cfg['role_plan_sha256'] == role_hashes(plans), 'Role plans/order hashes changed')
    require(sha(root/'provenance/build-manifest.json') == info['build_manifest']['sha256'], 'Build snapshot changed')
    for name, relative in (('parent-ablation4-manifest.json', 'parent-ablation4-manifest.json'),
                           ('parent-candidate-manifest.json', 'parent-candidate-manifest.json'),
                           ('parent-fused-manifest.json', 'parent-fused-manifest.json'),
                           ('parent-plans.json', 'parent-plans.json'), ('improved-manifest.json', 'improved/manifest.json')):
        require(sha(root/'provenance'/name) == manifest['files'][relative], 'Frozen build snapshot changed: '+name)
    builder.validate_snapshots(manifest, template, candidate, fused, improved, built=info['executable'], verify_evidence=not offline)
    builder.planner.validate(plans, verify_evidence=not offline)
    require(plans['parent'] == template['parent'], 'Execution plan changed original A provenance')
    require(info['selection']['path'] == 'provenance/selection/selection.json', 'Selection evidence path changed')
    selected_path = root/info['selection']['path']
    require(sha(selected_path) == seal['selection_sha256'] == info['selection']['sha256'] == info['selection_sha256'], 'Frozen probe selection changed')
    require(sha(selected_path.parent/'arrival-op2-manifest.json') == info['build_manifest']['sha256'],
            'Probe/build manifest changed')
    selected = probe.selection_verify(selected_path, selected_path.parent, offline=True)
    require(selected['selected_spans'] == options == info['selection']['selected_spans'] and
            selected['source_only'] == info['source_only'] and
            selected['gpu_measured'] == info['selection']['gpu_measured'] == info['executable'],
            'Probe selection or GPU evidence identity changed')
    require(manifest['source_only'] == plans['source_only'] == info['source_only'], 'Build rehearsal identity changed')
    budget_metadata(plans)
    if info['executable']:
        require(all(hardware_identity(r['hardware']) == hardware_identity(info['calibration_hardware'])
                    for r in plans['parent']['roles'].values()), 'Calibration hardware/order changed')
    require(set(info['variants']) == set(POLICIES), 'Missing/extra arrival_op2 policies')
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
    expected_spans = routing.IDENTITY_SPANS if policy == routing.A else info['schedule_options']
    require(all(r['communication'] == info['communication'] and r['losses'] == rows[0]['losses'] and
                r['execution_timing'] == routing.TIMING and
                r['schedule_options'] == info['schedule_options'] and r['runtime_spans'] == expected_spans
                for r in rows), 'TP loss/communication/timing/schedule mismatch')
    return rows


def profile_roles(events, policy):
    result = scope_report.profile_roles(events, policy, routing)
    for role, record in result.items():
        decoders = [name for name in record['kernels'] if 'taco_decode_ring_kernel' in name]
        require(decoders and all('taco_decode_ring_kernel_op2' not in name and
                                'warp' not in name.lower() for name in decoders),
                f'Both scheduling arms must execute the unchanged original CTA decoder: {policy}, {role}')
    return result


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
                selection_sha256=info['selection_sha256'],
                communication=info['communication'], fingerprints=reference, role_profiles=profiles,
                execution_timing=dict(routing.TIMING), files={str(p.relative_to(root)): sha(p) for p in files})
    options = builder.planner.frozen_options(info)
    if options is not None:
        gate['schedule_options'] = options
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


def acceptance(summary, nonidentity=True):
    """Independent S/A target; identity fallback cannot count as scheduling success."""
    require(type(nonidentity) is bool, 'Nonidentity flag must be Boolean')
    comparison = summary['comparisons'][routing.S+'/'+routing.A]
    interval = comparison['ci95']
    finite = lambda value: type(value) in (int, float) and math.isfinite(value)
    require(finite(comparison['percent']), 'Acceptance point must be a finite number')
    require(interval is None or (isinstance(interval, (list, tuple)) and len(interval) == 2 and
            all(finite(value) for value in interval) and interval[0] <= interval[1]),
            'Acceptance CI must contain two finite ordered numbers')
    passed = (nonidentity and comparison['percent'] >= routing.ACCEPTANCE_PERCENT and
              interval is not None and interval[0] >= routing.ACCEPTANCE_PERCENT)
    return dict(candidate=routing.S, reference=routing.A, schedule_changed=nonidentity,
                threshold_percent=routing.ACCEPTANCE_PERCENT,
                point_percent=comparison['percent'], ci95=interval,
                passed=passed, rule='nonidentity schedule and point and paired CI95 lower bound >= 1.5 percent',
                status='target_met_this_compare' if passed else 'target_not_demonstrated',
                independent_confirmation_required=True)


def markdown(result):
    lines = ['# H100 到达优先 op2：固定量化实现的调度替代实验', '',
             f'每窗口预热10步、正式50步；{result["timing_windows"]}个正式窗口，'
             f'{result["formal_optimizer_steps"]}个正式optimizer step。',
             'A保持消融7的交替＋全远端量化；S仅改变生产端tile调度，使用配对行局部性替代旧BF16到达拟合。',
             '两组使用逐字节相同的编码器、CTA解码器、全远端mask、单工作区、输出clone和两道屏障。',
             'attn/mlp跨度由独立同量化算子探测预先冻结，完整训练不再调参。',
             'S/A要求实际改变映射，且完整step耗时下降点估计及配对95%区间下界均≥1.5%。',
             '每窗口取8 rank最大elapsed/50，计入完整32层、4微批、8192 token的optimizer step。', '']
    for role, metadata in result['arrival_schedule'].items():
        lines.append(f'S的{role}跨度={result["schedule_options"][role]}，每rank改变槽位数：'
                     f'{metadata["changed_per_rank"]}；映射与A相同：{metadata["identical_to_A"]}。')
    for rep, summary in result['rounds'].items():
        lines += ['', f'## {rep}', '', '| 指标 | '+' | '.join(routing.LABELS[p] for p in POLICIES)+' |',
                  '| --- | '+' | '.join(['---:']*len(POLICIES))+' |']
        for field, label, fmt in (('mean_ms', '平均耗时（ms/step）', '.3f'),
                                  ('median_ms', '中位耗时（ms/step）', '.3f'),
                                  ('tokens_per_second', 'tokens/s', '.2f'),
                                  ('mean_loss', '平均训练loss', '.6f'),
                                  ('mean_final_loss', '最终训练loss平均', '.6e')):
            lines.append('| '+label+' | '+' | '.join(format(summary[field][p], fmt) for p in POLICIES)+' |')
        lines += ['', '| X / 参考Y | 配对耗时下降% [95% CI] |', '| --- | ---: |']
        for key, value in summary['comparisons'].items():
            a, b = key.split('/'); ci = value['ci95']
            interval = f'[{ci[0]:+.3f}, {ci[1]:+.3f}]' if ci else '不足两块'
            lines.append(f'| {routing.LABELS[a]} / {routing.LABELS[b]} | {value["percent"]:+.3f} {interval} |')
        lines.append(f'S/A的1.5%判据：{result["acceptance_by_round"][rep]["status"]}。')
    lines += ['', '每次比较以combined结果验收；后台批次要求两次独立完整比较均达标，逐轮结果完整保留。',
              '主表保留全部样本，按完整A/S配对块bootstrap10000次、seed20261009。',
              'loss只计rank0的正式step；固定合成微批短程训练不代表真实语料泛化。']
    for rep, sensitivity in result['sensitivity'].items():
        lines.append(f'慢窗口敏感性轮次{rep}：保留{sensitivity["remaining_blocks"]}块，排除：`{sensitivity["excluded"]}`。')
    return '\n'.join(lines)+'\n'


def report(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline)
    require(info['executable'] and not info['source_only'], 'CPU rehearsal cannot certify GPU execution')
    gate = read(root/'preflight.json')
    require(gate['completed'] and gate['protocol'] == PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json') and gate['communication'] == info['communication'] and
            gate['execution_timing'] == routing.TIMING and
            gate['selection_sha256'] == info['selection_sha256'], 'Fresh complete preflight required')
    options = builder.planner.frozen_options(info)
    require(builder.planner.frozen_options(gate) == options, 'Preflight schedule changed')
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
                  acceptance_threshold=routing.ACCEPTANCE_PERCENT, sample_policy='all samples primary', measurement_identity=measurement_identity(info, gate),
                  bootstrap=dict(resamples=10000, seed=20261009, unit='complete two-policy block', multiple_comparison_correction=None))
    if options is not None:
        result['schedule_options'] = options
    result['selection_sha256'] = info['selection']['sha256']
    result['measurement_identity']['communication'] = info['communication']
    combined = {p: [] for p in POLICIES}; all_blocks = []; files = {}
    for rep in range(1, info['rounds']+1):
        values = {p: [] for p in POLICIES}; blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]; samples = []
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == 2 and {j['policy'] for j in jobs} == set(POLICIES), 'Incomplete two-policy paired block')
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
    nonidentity = any(not value['identical_to_A'] for value in result['arrival_schedule'].values())
    result['acceptance_by_round'] = {key: acceptance(value, nonidentity) for key, value in result['rounds'].items()}
    result['acceptance'] = dict(result['acceptance_by_round']['combined'])
    result['arrival_diagnostic_only'] = False
    result['scheduling_method'] = 'destination-preserving-row-pair-v1'
    require(len(result['all_samples']) == info['timing_windows'], 'Missing timing samples')
    result.update(completed=True, timing_windows=info['timing_windows'], formal_optimizer_steps=50*info['timing_windows'],
                  timing_rank_records=8*info['timing_windows'], preflight_rank_records=32, profile_traces=16, candidate_shape_mappings=4,
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=files))
    write(root/'report.json', result); (root/'report.md').write_text(markdown(result))
    return result
