"""Three-arm identity, execution-path, numerical and paired timing evidence."""
from bisect import bisect_right
import gzip
import json
import math
from pathlib import Path
import statistics

import mav2_protocol as routing
import mav2_sources as sources
import ablation3_report
import ablation4_runtime as communication
import ablation6_build as flux_build
import mav2_build as codec_build
import mav2_validation as baseline_codec_validation
import scope_report
from mav2_prepare import SHARED_ENVIRONMENT, model, runtime_root, shape_metadata
from ablation_prepare import megatron_check
from ablation_report import measurement_identity
from common import HERE, MEGATRON_COMMIT, read, require, sha, verify_files, write
from report import fingerprints
from scope_plan import validate_hardware

PROTOCOL, POLICIES = routing.PROTOCOL, routing.POLICIES


def verify_inputs(root, offline=False):
    root = Path(root)
    info = read(root/'submission.json')
    require(info['protocol'] == PROTOCOL and info['config'] == model() and
            info['comparison_scope'] == 'both' and info['megatron_commit'] == MEGATRON_COMMIT,
            'Wrong MAV2 protocol/model/scope')
    require(type(info['source_only']) is bool and info['executable'] == (not info['source_only']),
            'Rehearsal identity changed')
    verify_files(root, info['files'])
    expected = routing.schedule(info['blocks_per_round'], info['schedule_seed'], info['rounds'])
    require(info['schedule'] == read(root/'schedule.json') == expected and
            info['timing_windows'] == len(expected), 'Incomplete or altered three-arm schedule')
    cfg = read(root/'scripts/config.json')
    seal = read(root/'provenance/preparation.json')
    require(all(seal[k] == info[k] for k in seal), 'Preparation identity changed')
    require(type(info.get('allow_quality_failure')) is bool and
            cfg.get('allow_quality_failure') is info['allow_quality_failure'] and
            seal.get('allow_quality_failure') is info['allow_quality_failure'],
            'Quality-failure policy changed; prepare a fresh comparison')
    require(info['execution_timing'] == routing.TIMING and all(type(cfg[k]) is int and cfg[k] == v for k, v in routing.TIMING.items()) and
            cfg['smoke_warmup_steps'] == 2 and cfg['smoke_steps'] == 3, 'Timing/smoke steps changed')
    require(info['quality_budget'] == cfg['quality_budget'] == routing.QUALITY_BUDGET and
            info['transform'] == cfg['transform'] == routing.TRANSFORM,
            'Pre-registered transformation or quality thresholds changed')
    require(cfg['protocol'] == PROTOCOL and cfg['policies'] == list(POLICIES) and cfg['repetitions'] == 2 and
            cfg['graph_forward'] is False and cfg['double_buffered'] is False and cfg['ring_reduction'] is True,
            'Fusion policy/lifecycle changed')
    m = model()
    require(cfg['common'] == {k: m[k] for k in ('layers', 'micro_batch', 'global_batch', 'seed', 'token_seed', 'vocab')} and
            cfg['cases'] == [dict(name='h100-phase6', **{k: m[k] for k in ('hidden', 'ffn', 'heads', 'sequence', 'tp')})],
            'Worker model changed')
    require(read(root/'scripts/selection-plan.json') == shape_metadata() and
            cfg['role_plan_sha256'] == {p: {r: None for r in routing.SHAPES} for p in POLICIES},
            'Arrival/selection plans are forbidden')
    require(info['shared_environment'] == SHARED_ENVIRONMENT and
            info['communication'] == cfg['communication'] == communication.validate(info['communication']),
            'Communication/submission environment changed')
    require(info['hardware'] == info['calibration_hardware'] and
            info['hardware_identity_kind'] == 'runtime identity only; no arrival calibration',
            'Hardware identity changed')
    if info['executable']:
        validate_hardware(info['hardware'])
    else:
        require(info['hardware'] == {}, 'CPU rehearsal must not claim GPU identity')
    require(info['codec_identity'] == cfg['codec_identity'] == sources.codec_identity(), 'Standalone codec source changed')
    for name, expected_source in {'adapter.py': sources.adapter_source(), 'worker.py': sources.worker_source(info['source_only']),
                                  'taco_support.py': sources.taco_source(), 'megatron_taco.py': sources.codec_source()}.items():
        require((root/'scripts'/name).read_text() == expected_source, 'Frozen execution source changed: '+name)
    require(sha(root/'scripts/routing.py') == sha(HERE/'mav2_protocol.py') and
            sha(root/'scripts/reference.py') == info['codec_identity']['reference_sha256'] and
            sha(root/'scripts/mav2_reference.py') == info['codec_identity']['selective_reference_sha256'],
            'Routing/reference changed')
    snapshots = {'flux_build_manifest': 'gemm-only-manifest.json', 'codec_build_manifest': 'codec-manifest.json'}
    for key, filename in snapshots.items():
        require(info[key]['sha256'] == sha(root/'provenance'/filename), 'Frozen build snapshot changed: '+key)
    gemm = read(root/'provenance/gemm-only-manifest.json')
    codec = read(root/'provenance/codec-manifest.json')
    flux_build.validate_snapshot(gemm, built=info['executable'])
    codec_build.validate_snapshot(codec, built=info['executable'])
    require(gemm['source_only'] == codec['source_only'] == info['source_only'] and
            info['codec_library'] == cfg['codec_library'] == codec['library'], 'Build/codec rehearsal mode changed')
    flux_root = Path(info['flux_build_manifest']['path']).parent
    require(set(info['variants']) == set(POLICIES), 'Missing/extra arms')
    for p in POLICIES:
        v = info['variants'][p]
        require(all(v.get(k) == value for k, value in routing.definition(p).items()) and v['sampling'] is False and
                Path(v['root']) == runtime_root(flux_root, p), 'Policy implementation/runtime root changed')
        if info['executable']:
            require(set(v['libraries']) == {'libflux_cuda.so', 'libflux_cuda_ths_op.so'}, 'Missing runtime libraries')
            manifest = gemm
            prefix = flux_build.VARIANT
            for name, identity in v['libraries'].items():
                require(identity['sha256'] == manifest['files'][prefix+'/'+name], 'Library differs from build manifest')
                if not offline:
                    require(Path(identity['path']) == (Path(v['root'])/name).resolve() and sha(identity['path']) == identity['sha256'],
                            'Loaded runtime binary changed')
        else:
            require(v['libraries'] == {}, 'CPU rehearsal cannot register executable libraries')
    require(all(info['variants']['basic']['root'] == info['variants'][p]['root'] and
                info['variants']['basic']['libraries'] == info['variants'][p]['libraries']
                for p in POLICIES), 'All arms must share the exact GemmOnly build')
    for src, dest in (('ablation6_flux_tensor.py', 'flux_tensor_taco.py'),
                      ('mav2_validation.py', 'mav2_validation.py'),
                      ('basic_quant_validation.py', 'basic_quant_validation.py'),
                      ('mav2_reference.py', 'mav2_reference.py'),
                      ('baseline_codec_validation.py', 'baseline_codec_validation.py')):
        require(sha(root/'scripts'/dest) == sha(HERE/src), 'Frozen shared execution source changed: '+dest)
    if not offline:
        verify_files('/', info['controllers'])
        for key in snapshots:
            require(sha(info[key]['path']) == info[key]['sha256'], 'Live build manifest changed')
        flux_build.verify(flux_root, built=info['executable'])
        codec_build.verify(Path(info['codec_build_manifest']['path']).parent, built=info['executable'])
        if info['executable']:
            megatron_check(Path(info['megatron']))
    return info


def window(root, folder, policy, mode, block=-1, index=-1):
    root, folder = Path(root), Path(folder)
    rows = scope_report.window(root, folder, policy, mode, block, index, routing)
    info, command = read(root/'submission.json'), read(folder/'command.json')
    require(command['communication'] == info['communication'] and command['scope'] == routing.definition(policy) and
            all(command[k] == v for k, v in dict(policy=policy, mode=mode, block=block, index=index).items()),
            'Worker command identity changed')
    for row in rows:
        require(row['communication'] == info['communication'] and row['shared_environment'] == SHARED_ENVIRONMENT and
                row['losses'] == rows[0]['losses'], 'TP trajectory/environment mismatch')
        require(row['codec_identity'] == info['codec_identity'], 'Wrong standalone codec definition')
        expected = {n: (row['warmup_steps']+len(row['losses']))*4 for n in row['enabled_modules']}
        require(row['codec_calls'] == expected and
                row['codec_library'] == info['codec_library'],
                'Standalone CUDA codec route/call count changed')
        diagnostics = row.get('transform_diagnostics', {})
        if mode in ('smoke', 'profile'):
            require(set(diagnostics) == set(row['enabled_modules']), 'Missing real-model transform samples')
            for name, d in diagnostics.items():
                require(d['policy'] == policy and d['calls'] == expected[name] and d['sampled_calls'] == 1 and
                        d['total'] == 7*256*32 and type(d['d8']) is int and type(d['h8']) is int and
                        d['d8'] >= 0 and d['h8'] >= 0 and d['total'] == d['d8']+d['h8'] and
                        d['nonfinite'] == d['invalid_tags'] == 0 and d['bytes_per_group'] == 136 and
                        d['selector'] == {'relative_l2': .025, 'max_abs_over_amax': .0625} and
                        d['quality_selection'] is (policy == 'mav2') and
                        d['h8_fallback_allowed'] is (policy == 'mav2') and
                        d['quantized_remote_fraction'] == 1.0 and d['d8_fraction'] == d['d8']/d['total'],
                        'Invalid real-model D8/H8 transform evidence')
                require(policy != 'basic' or d['d8'] == 0, 'H8 control used selective transformation')
                require(policy != 'd8' or d['d8'] == d['total'] and d['h8'] == 0,
                        'Forced-D8 arm executed H8 or selective transformation')
        else:
            require(diagnostics == {}, 'Transform diagnostics leaked into formal timing')
        if mode == 'smoke':
            require(set(row['smoke_output_sha256']) == set(row['enabled_modules']) and
                    all(isinstance(h, str) and len(h) == 64 and all(x in '0123456789abcdef' for x in h)
                        for h in row['smoke_output_sha256'].values()), 'Missing initial projection output fingerprints')
            require(set(row['retained_output_checks']) == set(routing.SHAPES), 'Both retained-output roles are required')
            for role, check in row['retained_output_checks'].items():
                require(check['module'] in row['enabled_modules'] and routing.module_role(check['module']) == role and
                        check['outputs'] == 6 and check['independent_storage'] is True and
                        check['unchanged_after_subsequent_call'] is True and check['passed'] is True,
                        'Retained outputs were not independently owned or survived reuse')
            baseline_codec_validation.verify(row['codec_validation'], info['codec_library'], row['rank'])
        else:
            require(row['codec_validation'] is None and not row['smoke_output_sha256'] and not row['retained_output_checks'],
                    'Smoke diagnostics leaked into timing/profile')
    return rows


def quality_pair(basic, candidate):
    """Screen real training trajectories; D8 is allowed to change rounding.

    Ratios are descriptive (None at zero control); the declared absolute-plus-
    relative budget is the gate. Preserve all diagnostics, including small-loss
    ratios and negative performance results, instead of claiming equivalence.
    """
    require(len(basic) == len(candidate) == 8, 'Eight ranks are required for quality comparison')
    result = {'passed': True, 'budget': dict(routing.QUALITY_BUDGET), 'ranks': []}
    for rank, (a, b) in enumerate(zip(basic, candidate)):
        fields = {}
        for field in ('losses', 'grad_norms', 'initial_loss', 'warmup_last_loss'):
            left = a[field] if isinstance(a[field], list) else [a[field]]
            right = b[field] if isinstance(b[field], list) else [b[field]]
            require(len(left) == len(right) and len(left) > 0 and
                    all(math.isfinite(x) for x in left+right), 'Incomplete/nonfinite quality trajectory')
            key = 'grad_norm' if field == 'grad_norms' else 'loss'
            atol, rtol = (routing.QUALITY_BUDGET[key+'_'+suffix] for suffix in ('atol', 'rtol'))
            errors = [abs(x-y) for x, y in zip(left, right)]
            scores = [e/(atol+rtol*abs(x)) for e, x in zip(errors, left)]
            rms = math.sqrt(sum(e*e for e in errors)/len(errors))
            scale = math.sqrt(sum(x*x for x in left)/len(left))
            worst = max(range(len(scores)), key=scores.__getitem__)
            fields[field] = dict(passed=max(scores) <= 1, max_budget_fraction=max(scores),
                                 failed_points=sum(score > 1 for score in scores),
                                 worst_point=dict(index=worst, control=left[worst], candidate=right[worst],
                                     abs_error=errors[worst], allowed_abs_error=atol+rtol*abs(left[worst])),
                                 max_abs=max(errors), rmse=rms,
                                 normalized_rmse=rms/scale if scale else None,
                                 control_mean=statistics.mean(left), candidate_mean=statistics.mean(right),
                                 mean_ratio=statistics.mean(right)/statistics.mean(left) if statistics.mean(left) else None,
                                 control_final=left[-1], candidate_final=right[-1],
                                 final_ratio=right[-1]/left[-1] if left[-1] else None)
        passed = all(value['passed'] for value in fields.values())
        result['ranks'].append(dict(rank=rank, passed=passed, fields=fields))
        result['passed'] = result['passed'] and passed
    return result


def quality_failure_summary(quality):
    """One worst rank/point per failed arm, mode and metric."""
    failures = []
    for policy, modes in quality.items():
        for mode, pair in modes.items():
            fields = pair['ranks'][0]['fields']
            for field in fields:
                rank = max(pair['ranks'], key=lambda r: r['fields'][field]['max_budget_fraction'])
                value = rank['fields'][field]
                if value['passed']:
                    continue
                point = value['worst_point']
                failures.append(f'{policy}/{mode}/{field}: rank={rank["rank"]}, index={point["index"]}, '
                    f'basic={point["control"]:.6g}, candidate={point["candidate"]:.6g}, '
                    f'error={point["abs_error"]:.6g} > budget={point["allowed_abs_error"]:.6g} '
                    f'({value["max_budget_fraction"]:.3f}x)')
    return '; '.join(failures)


def admit_preflight(root, info, gate):
    """Only a sealed opt-in may admit finite training-quality failures."""
    require(gate['completed'] is True and gate['execution_path_validated'] is True and
            gate['submission_sha256'] == sha(Path(root)/'submission.json'),
            'Complete passing numerical/path preflight required')
    require(gate.get('allow_quality_failure') is info['allow_quality_failure'] and
            type(gate['correctness_passed']) is bool and
            gate['correctness_passed'] is gate['quality_within_pairs'] and
            set(gate['quality_by_candidate']) == set(POLICIES)-{'basic'} and
            all(type(value) is bool for value in gate['quality_by_candidate'].values()) and
            gate['quality_within_pairs'] == all(gate['quality_by_candidate'].values()),
            'Preflight quality status/policy mismatch')
    require(gate['quality_within_pairs'] or info['allow_quality_failure'],
            'Complete passing numerical/path preflight required; training quality failed')


def contained(child, parent):
    return (child.get('pid'), child.get('tid')) == (parent.get('pid'), parent.get('tid')) and \
        child['ts'] >= parent['ts'] and child['ts']+child.get('dur', 0) <= parent['ts']+parent['dur']+.01


def profile_roles(events, policy):
    """Associate actual kernels by CPU launch correlation, ignoring mirrored GPU annotations."""
    routing.definition(policy)
    launches = [e for e in events if e.get('cat') in ('cuda_runtime', 'cuda_driver') and
                e.get('args', {}).get('correlation') is not None]
    kernels = [e for e in events if e.get('cat') == 'kernel']
    cpu_ops = [e for e in events if e.get('cat') == 'cpu_op']
    by_thread = {}
    for e in launches:
        by_thread.setdefault((e.get('pid'), e.get('tid')), []).append(e)
    for group in by_thread.values():
        group.sort(key=lambda e: e['ts'])
    starts = {key: [e['ts'] for e in group] for key, group in by_thread.items()}
    kernel_by_correlation = {}
    for e in kernels:
        kernel_by_correlation.setdefault(e.get('args', {}).get('correlation'), []).append(e)
    result, claimed = {}, set()
    for role in routing.SHAPES:
        route = 'flux_tensor'
        scopes = [e for e in events if e.get('cat') == 'user_annotation' and e.get('ph') == 'X' and
                  e.get('name') == f'scope.{role}.{route}']
        require(len(scopes) == 128, 'Missing/duplicated real CPU role annotations')
        values = []
        for scope in scopes:
            thread = (scope.get('pid'), scope.get('tid'))
            lo = bisect_right(starts.get(thread, []), scope['ts'])-1
            call_launches = [e for e in by_thread.get(thread, [])[max(lo, 0):]
                             if e['ts'] <= scope['ts']+scope['dur'] and contained(e, scope)]
            path = [e for launch in call_launches for e in kernel_by_correlation.get(launch['args']['correlation'], [])]
            require(len({id(e) for e in path}) == len(path) and not (claimed & {id(e) for e in path}),
                    'A GPU kernel was attributed twice')
            claimed.update(id(e) for e in path)
            names = [e.get('name', '') for e in path]
            barriers = sorted([e for e in path if 'CudaIpcBarrierAllKernel' in e.get('name', '')], key=lambda e: e['ts'])
            clones = [e for e in cpu_ops if e.get('name') == 'aten::clone' and contained(e, scope)]
            gemms = [e for e in path if ('cutlass' in e['name'].lower() or 'flux_bf16' in e['name'].lower()) and
                     'sm90' in e['name'].lower()]
            encoder = 'fused_ash_compress_kernel' if policy == 'basic' else 'mav2_encode_kernel'
            enc = [e for e in path if encoder in e['name']]
            receiver = 'basic_quant_decode_reduce_kernel' if policy == 'basic' else 'mav2_decode_reduce_kernel'
            dec = [e for e in path if receiver in e['name']]
            wrong = ('mav2_encode_kernel', 'mav2_decode_reduce_kernel') if policy == 'basic' else (
                'fused_ash_compress_kernel', 'basic_quant_decode_reduce_kernel')
            require(not any(token in n for n in names for token in wrong) and
                    not any('fused_ash_decompress_kernel' in n or 'mav2_decode_kernel' in n for n in names),
                    'Encoder/receiver path was substituted')
            adds = [e for e in cpu_ops if e.get('name') == 'aten::add' and contained(e, scope)]
            require(len(adds) == 0, 'All arms require fused ordered BF16 reduction')
            require(len(gemms) == len(enc) == len(dec) == 1 and not barriers and not clones and
                    not any('taco_decode_ring_kernel' in n or 'taco_basic_' in n or
                            'reducescatter' in n.lower().replace('_', '') for n in names),
                    'All arms require GemmOnly, declared receiver and standalone encoder, without GEMM-RS/clone')
            require(any('nccl' in n.lower() for n in names), 'Standalone exchange lacks actual NCCL kernels')
            markers = [e for e in events if e.get('cat') == 'user_annotation' and contained(e, scope)]
            for name in ('ablation6.flux_gemm_only', 'megatron_taco.encode', 'megatron_taco.exchange', 'megatron_taco.decode_reduce'):
                require(sum(e.get('name') == name for e in markers) == 1, 'Missing standalone stage: '+name)
            require(any('alltoall' in e.get('name', '').lower().replace('_', '') and contained(e, scope) for e in cpu_ops),
                    'Standalone NCCL All-to-All call missing')
            values.append(dict(gemm_us=gemms[0]['dur'], decode_us=dec[0]['dur'], encode_us=enc[0]['dur']))
        result[role] = dict(calls=128, route=route, barriers=0,
                            output_clones=0,
                            receiver_kernel_calls=128, separate_bf16_adds=0,
                            receiver_metric='decode+BF16 fold kernel',
                            encoder_median_us=statistics.median(x['encode_us'] for x in values),
                            gemm_median_us=statistics.median(x['gemm_us'] for x in values),
                            receiver_median_us=statistics.median(x['decode_us'] for x in values))
    return result


def preflight(root, offline=False):
    root = Path(root)
    info = verify_inputs(root, offline)
    require(info['executable'] and not info['source_only'], 'CPU rehearsals cannot certify GPU execution')
    start = read(root/'start-gpu-check.json')
    require(start['passed'] is True and start['run_id'] == info['run_id'], 'Missing fresh startup check')
    files, reference, profiles, records = [root/'start-gpu-check.json'], None, {}, {}
    for p in POLICIES:
        for mode in ('smoke', 'profile'):
            folder = root/'results'/f'{mode}-{p}'
            rows = window(root, folder, p, mode)
            records[(p, mode)] = rows
            key = fingerprints(rows)
            if reference is None:
                reference = key
            require(key == reference, 'Cross-arm initialization/data/software/hardware mismatch')
            files += [f for f in folder.rglob('*') if f.is_file()]
        for rank in range(8):
            with gzip.open(root/'results'/f'profile-{p}'/f'rank{rank}-profile.json.gz', 'rt') as handle:
                profiles[f'{p}/rank{rank}'] = profile_roles(json.load(handle)['traceEvents'], p)
    quality = {policy: {mode: quality_pair(records[('basic', mode)], records[(policy, mode)])
                        for mode in ('smoke', 'profile')} for policy in POLICIES if policy != 'basic'}
    quality_by_candidate = {policy: all(item['passed'] for item in modes.values())
                            for policy, modes in quality.items()}
    passed = all(quality_by_candidate.values())
    transforms = {p: {mode: {str(row['rank']): row['transform_diagnostics'] for row in records[(p, mode)]}
                      for mode in ('smoke', 'profile')} for p in POLICIES}
    gate = dict(protocol=PROTOCOL, completed=True, correctness_passed=passed, execution_path_validated=True,
                allow_quality_failure=info['allow_quality_failure'],
                bitwise_equal_within_pairs=False, quality_within_pairs=passed,
                quality_by_candidate=quality_by_candidate,
                quality_budget=dict(routing.QUALITY_BUDGET), quality=quality,
                equivalence_pairs=routing.EQUIVALENCE_PAIRS,
                submission_sha256=sha(root/'submission.json'), communication=info['communication'],
                codec_identity=info['codec_identity'], codec_library=info['codec_library'], fingerprints=reference,
                role_profiles=profiles, files={str(p.relative_to(root)): sha(p) for p in files})
    gate['transform_diagnostics'] = transforms
    gate['quality_failure_summary'] = quality_failure_summary(quality)
    write(root/'preflight.json', gate)
    if not passed:
        message = ('MAV2 preflight training-quality budget failed; diagnostics saved in preflight.json: '
                   + gate['quality_failure_summary'])
        require(info['allow_quality_failure'], message)
        print('QUALITY FAILED; continuing performance measurement by explicit opt-in. '+message, flush=True)
    return gate


def summarize(values, blocks, samples):
    result = ablation3_report.summarize(values, blocks, POLICIES, routing.COMPARISONS)
    result['mean_ms'] = {p: statistics.mean(v) for p, v in values.items()}
    result['tokens_per_second'] = {p: 8192000/ms for p, ms in result['mean_ms'].items()}
    result['mean_loss'] = {p: statistics.mean(s['mean_loss'] for s in samples if s['policy'] == p) for p in POLICIES}
    result['mean_final_loss'] = {p: statistics.mean(s['final_loss'] for s in samples if s['policy'] == p) for p in POLICIES}
    return result


def summarize_transforms(diagnostics):
    result = {}
    for policy in POLICIES:
        result[policy] = {}
        for role in routing.SHAPES:
            rows = [value for modules in diagnostics[policy]['smoke'].values()
                    for name, value in modules.items() if routing.module_role(name) == role]
            direct, rotated = (sum(row[key] for row in rows) for key in ('d8', 'h8'))
            result[policy][role] = dict(samples=len(rows), d8=direct, h8=rotated,
                                       total=direct+rotated, d8_fraction=direct/(direct+rotated))
    return result


def markdown(result):
    lines = ['# MAV2：基础量化、选择性变换与强制 D8 的同场完整训练', '',
             f'本次 {result["timing_windows"]} 个正式窗口、{result["formal_optimizer_steps"]} 个正式 optimizer step；每窗口 10 步预热＋50 步计时。',
             '三组均覆盖 Attention＋MLP、固定全远端量化、136 B 包、同一 GEMM 和 NCCL。basic 全 H8；mav2 逐组选择 D8/H8；d8 强制全部直接量化且不运行选择器。',
             '每个候选独立对照 basic：逐点 loss 门槛为 0.05+5%×对照绝对值，grad_norm 为 0.1+10%×对照绝对值。',
             f'短程数值门禁通过：{result["quality_within_pairs"]}。逐 rank/逐窗口误差、均值/末步比与归一化 RMSE 保存在 JSON quality 字段。',
             '三组没有预设耗时排序；负结果全部保留。']
    if result['allow_quality_failure']:
        lines += ['', '本批次显式允许质量超限后继续性能测量；数值门槛未放宽，预检与正式窗口的质量失败均计入最终判定。']
    if not result['quality_within_pairs']:
        lines += ['', '**质量门禁未通过：流程完成不代表数值合格；质量失败的比较不能认定为合格加速。**']
    lines += ['', '| 候选 | 相对 basic 的数值门禁 | 耗时下降点估计为正 | 正收益且 95% CI 下界为正 |',
              '| --- | --- | --- | --- |']
    for policy, acceptance in result['candidate_acceptance'].items():
        lines.append(f'| {policy} | {acceptance["quality_passed"]} | {acceptance["positive_point_estimate"]} | '
                     f'{acceptance["benefit_confirmed"]} |')
    lines += ['', '变换比例来自 smoke 中每个 rank、每层投影的首个真实发送包；不是完整训练覆盖率。', '',
              '| 组别 | 角色 | D8 组数 | H8 组数 | D8 占比 |',
              '| --- | --- | ---: | ---: | ---: |']
    for policy, roles in result['transform_summary'].items():
        for role, values in roles.items():
            lines.append(f'| {policy} | {role} | {values["d8"]} | {values["h8"]} | {values["d8_fraction"]:.3%} |')
    for rep, row in result['rounds'].items():
        lines += ['', f'## {rep}', '', '| 指标 | '+' | '.join(routing.LABELS[p] for p in POLICIES)+' |',
                  '| --- |'+' ---: |'*len(POLICIES)]
        for field, label, fmt in [('mean_ms', '平均 ms/step', '.3f'), ('median_ms', '中位 ms/step', '.3f'),
                                  ('mean_loss', '平均 loss', '.6f'), ('mean_final_loss', '末步 loss 平均', '.6e')]:
            lines.append('| '+label+' | '+' | '.join(format(row[field][p], fmt) for p in POLICIES)+' |')
        lines += ['', '| 比较 | 配对耗时下降 %［95% CI］ |', '| --- | ---: |']
        for key, stat in row['comparisons'].items():
            lines.append(f'| {key} | {stat["percent"]:+.3f} [{stat["ci95"][0]:+.3f}, {stat["ci95"][1]:+.3f}] |')
    lines += ['', '每个收益结论仅在该比较涉及的候选均通过相对 basic 的质量门禁时有效。',
              '主结果保留全部完整块；按配对块 bootstrap 10000 次，未做多重比较校正。',
              '独立窗口累计训练步数不等于一条连续训练轨迹；固定合成数据不代替真实语料质量验证。']
    return '\n'.join(lines)+'\n'


def report(root, offline=False, save=True):
    root = Path(root)
    info = verify_inputs(root, offline)
    gate = read(root/'preflight.json')
    admit_preflight(root, info, gate)
    require(info['executable'] and not info['source_only'] and
            gate['bitwise_equal_within_pairs'] is False and gate['quality_budget'] == routing.QUALITY_BUDGET and
            gate['equivalence_pairs'] == routing.EQUIVALENCE_PAIRS and gate['protocol'] == PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json') and
            all(gate[k] == info[k] for k in ('communication', 'codec_identity', 'codec_library')) and
            set(gate['quality_by_candidate']) == set(POLICIES)-{'basic'},
            'Fresh complete three-arm preflight is required')
    verify_files(root, gate['files'])
    require({str(p.relative_to(root/'results')) for p in (root/'results').glob('round*/window-*') if p.is_dir()} ==
            {j['directory'] for j in info['schedule']}, 'Missing/extra formal windows')
    result = dict(protocol=PROTOCOL, run_id=info['run_id'], completed=False, correctness_passed=True,
                  execution_path_validated=True, offline=offline, comparison_scope='both',
                  bitwise_equal_within_pairs=False, identical_trajectories_within_pairs=False,
                  allow_quality_failure=info['allow_quality_failure'],
                  quality_within_pairs=True, quality_budget=dict(routing.QUALITY_BUDGET), quality=[],
                  preflight_quality_by_candidate=dict(gate['quality_by_candidate']),
                  quality_by_candidate=dict(gate['quality_by_candidate']),
                  equivalence_pairs=routing.EQUIVALENCE_PAIRS, full_optimizer_steps_measured=True,
                  round_count=2, blocks_per_round=info['blocks_per_round'], config=info['config'],
                  execution_timing=dict(routing.TIMING), communication=info['communication'],
                  codec_identity=info['codec_identity'], codec_library=info['codec_library'],
                  measurement_identity=measurement_identity(info, gate), rounds={}, all_samples=[],
                  sample_policy='all complete blocks; no post-hoc exclusions', acceptance_threshold=None,
                  transform=dict(routing.TRANSFORM), primary_comparison='mav2/basic')
    result['transform_diagnostics'] = gate['transform_diagnostics']
    result['transform_summary'] = summarize_transforms(gate['transform_diagnostics'])
    combined, all_blocks, files = {p: [] for p in POLICIES}, [], {}
    for rep in (1, 2):
        values, samples = {p: [] for p in POLICIES}, []
        blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == len(POLICIES) and {j['policy'] for j in jobs} == set(POLICIES),
                    'Incomplete three-arm pairing block')
            paired_rows = {}
            for job in jobs:
                folder = root/'results'/job['directory']
                rows = window(root, folder, job['policy'], 'timing', job['block'], job['index'])
                paired_rows[job['policy']] = rows
                require(fingerprints(rows) == [tuple(x) for x in gate['fingerprints']], 'Formal window identity changed')
                elapsed = [r['elapsed_seconds'] for r in rows]
                ms = max(elapsed)*1000/routing.TIMING['timed_steps']
                values[job['policy']].append(ms)
                samples.append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms, tokens_per_second=8192000/ms,
                                    mean_loss=statistics.mean(rows[0]['losses']), final_loss=rows[0]['losses'][-1]))
                files.update({str(p.relative_to(root)): sha(p) for p in folder.rglob('*') if p.is_file()})
            for policy in result['quality_by_candidate']:
                quality = quality_pair(paired_rows['basic'], paired_rows[policy])
                result['quality'].append(dict(round=rep, block=block['block'], candidate=policy, **quality))
                result['quality_by_candidate'][policy] = result['quality_by_candidate'][policy] and quality['passed']
        result['rounds'][str(rep)] = summarize(values, blocks, samples)
        for p in POLICIES:
            combined[p] += values[p]
        all_blocks += blocks
        result['all_samples'] += samples
    result['rounds']['combined'] = summarize(combined, all_blocks, result['all_samples'])
    estimates = result['rounds']['combined']['comparisons']
    result['quality_within_pairs'] = all(result['quality_by_candidate'].values())
    result['comparison_acceptance'] = {}
    for candidate, control in routing.COMPARISONS:
        key = candidate+'/'+control
        stat = estimates[key]
        quality_passed = all(p == 'basic' or result['quality_by_candidate'][p] for p in (candidate, control))
        result['comparison_acceptance'][key] = dict(quality_passed=quality_passed,
            positive_point_estimate=stat['percent'] > 0,
            benefit_confirmed=quality_passed and stat['percent'] > 0 and stat['ci95'][0] > 0)
    result['candidate_acceptance'] = {p: dict(comparison=p+'/basic', **result['comparison_acceptance'][p+'/basic'])
                                      for p in result['quality_by_candidate']}
    adjacent = [estimates['mav2/basic']]
    result.update(completed=True, timing_windows=info['timing_windows'],
                  formal_optimizer_steps=routing.TIMING['timed_steps']*info['timing_windows'],
                  timing_rank_records=8*info['timing_windows'], preflight_rank_records=8*2*len(POLICIES),
                  profile_traces=8*len(POLICIES), candidate_shape_mappings=0,
                  expected_order_point_estimate=all(s['percent'] > 0 for s in adjacent),
                  correctness_passed=result['quality_within_pairs'],
                  expected_order_confirmed=result['quality_by_candidate']['mav2'] and
                    all(s['percent'] > 0 and s['ci95'][0] > 0 for s in adjacent),
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=files))
    if save:
        write(root/'report.json', result)
        (root/'report.md').write_text(markdown(result))
    return result
