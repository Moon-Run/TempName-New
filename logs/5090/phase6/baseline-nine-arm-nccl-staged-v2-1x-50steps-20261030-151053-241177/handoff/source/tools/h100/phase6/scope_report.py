"""Strict scope/shape/runtime evidence and same-scope paired full-step reports."""
import csv
from bisect import bisect_right
import gzip
import json
import math
from pathlib import Path
import statistics
import scope_protocol

from ablation_plan import mask_sha
from ablation2_report import native_profile
from ablation_report import measurement_identity
from common import read, require, sha, verify_files, write
from report import fingerprints, paired
from scope_protocol import PROTOCOL, POLICIES, SHAPES, definition, is_native, module_role, module_routes, schedule
from scope_plan import model_config, validate, hardware_identity

PRIMARY = [('interleaved_full_attn', 'native_taco_attn'),
           ('interleaved_full_attn_mlp', 'native_taco_attn_mlp')]
COMPARISONS = PRIMARY + [('interleaved_full_attn_mlp', 'interleaved_full_attn'),
                         ('native_taco_attn_mlp', 'native_taco_attn')]


def verify_inputs(root, offline=False):
    root = Path(root); info = read(root/'submission.json')
    require(info['protocol'] == PROTOCOL and info['config'] == model_config(), 'Wrong scope protocol/model')
    verify_files(root, info['files'])
    require(info['schedule'] == read(root/'schedule.json') == schedule(info['blocks_per_round'], info['schedule_seed']) and
            info['timing_windows'] == len(info['schedule']), 'Schedule changed')
    seal = read(root/'provenance/preparation.json')
    require(all(seal[k] == info[k] for k in ('protocol', 'run_id', 'source_only', 'executable')) and
            info['executable'] == (not info['source_only']), 'Rehearsal mode changed; prepare a new session')
    plans = read(root/'scripts/selection-plan.json'); cfg = read(root/'scripts/config.json')
    require(sha(root/'scripts/selection-plan.json') == seal['plans_sha256'] and
            cfg['protocol'] == PROTOCOL and cfg['policies'] == list(POLICIES) and set(plans['roles']) == set(SHAPES), 'Scope roles/plans changed')
    for role, record in plans['roles'].items():
        validate(record, role, allow_test=info['source_only'], verify_evidence=not offline)
        require(cfg['role_plan_sha256'][role] == record['entry_sha256'], 'Worker role hash changed')
    if not info['source_only']:
        require(not plans['source_only'] and plans['gpu_calibrated'] and
                all(hardware_identity(r['hardware']) == hardware_identity(info['calibration_hardware'])
                    for r in plans['roles'].values()), 'Real calibrated plans on identical devices required')
    m = read(root/'provenance/build-manifest.json')
    require(m['protocol'] == PROTOCOL and m['kind'] == 'scope_candidate' and
            sha(root/'provenance/build-manifest.json') == info['build_manifest']['sha256'], 'Candidate build identity changed')
    v = m['variants']['scope-candidate']
    require(v['role'] == 'candidate' and not v['sampling'], 'Sampler cannot be timed')
    require(set(info['variants']) == set(POLICIES), 'Missing/extra scopes')
    candidates = []
    for policy in POLICIES:
        actual = info['variants'][policy]
        require(all(actual.get(k) == value for k, value in definition(policy).items()), 'Scope definition changed')
        if is_native(policy):
            require(not any(k in actual for k in ('root', 'libraries', 'sampling')), 'Native scope uses a Flux build')
        else:
            candidates.append(actual)
            require(actual['sampling'] is False, 'Sampling library cannot be timed')
            if info['executable']:
                require(m['status'] == 'built' and not m['source_only'] and not m['diagnostic_only'] and
                        set(actual['libraries']) == {'libflux_cuda.so', 'libflux_cuda_ths_op.so'}, 'Unbuilt/diagnostic candidate')
                if not offline:
                    for lib in actual['libraries'].values():
                        require(sha(lib['path']) == lib['sha256'], 'Runtime library changed')
    require(candidates[0]['root'] == candidates[1]['root'] and candidates[0]['libraries'] == candidates[1]['libraries'],
            'Both candidate scopes must share the same dual-shape library')
    if not offline:
        require(sha(info['build_manifest']['path']) == info['build_manifest']['sha256'], 'Build manifest changed')
        verify_files('/', info['build_files'])
    return info


def window(root, folder, policy, mode, block=-1, index=-1, routing=scope_protocol):
    PROTOCOL, SHAPES = routing.PROTOCOL, routing.SHAPES
    definition, is_native = routing.definition, routing.is_native
    fused_v2 = getattr(routing, 'is_fused_v2', lambda p: False)(policy)
    flux_tensor = getattr(routing, 'is_flux_tensor', lambda p: False)(policy)
    module_role, module_routes = routing.module_role, routing.module_routes
    root, folder = Path(root), Path(folder); info = read(root/'submission.json')
    c = info['config']; cfg = read(root/'scripts/config.json'); plans = read(root/'scripts/selection-plan.json')
    role_records = routing.role_plans(plans, policy)
    model = dict(cfg['common'], **cfg['cases'][0]); v = info['variants'][policy]
    count = cfg['smoke_steps'] if mode == 'smoke' else 1 if mode == 'profile' else cfg['timed_steps']
    warmup = cfg['smoke_warmup_steps'] if mode == 'smoke' else cfg['warmup_steps']
    rows = [read(folder/f'rank{r}.json') for r in range(8)]
    for rank, r in enumerate(rows):
        require(r['completed'] and r['passed'] and r['skips'] == r['fallback_calls'] == 0, 'Failed/skipped/fallback window')
        require(r['protocol'] == PROTOCOL and r['run_id'] == info['run_id'] and r['policy'] == policy and r['mode'] == mode and
                r['scope_definition'] == definition(policy) and r['role_plan_sha256'] == cfg['role_plan_sha256'], 'Scope/role plan identity mismatch')
        require(r['rank'] == r['local_rank'] == r['tp_rank'] == rank and r['world_size'] == 8 and r['dp_rank'] == 0 and
                r['tp_group_ranks'] == list(range(8)) and r['dp_group_ranks'] == [rank], 'Missing local TP8 ranks')
        require(hardware_identity(r['hardware']) == hardware_identity(info['calibration_hardware']) and
                r['hardware']['local_rank'] == rank and r['gpu_uuid'] == r['hardware']['devices'][rank]['uuid'], 'Hardware/order changed')
        require(r['model'] == model and r['padded_vocab_size'] == 33792 and r['block'] == block and r['window'] == index,
                'Model/full shape/pairing mismatch')
        require(r['timing_protocol'] == 'full-step-one-clear-v2' and not r['sampling'] and r['profiling'] == (mode == 'profile'), 'Wrong training path')
        require(r['warmup_steps'] == warmup and len(r['losses']) == len(r['grad_norms']) == count and
                all(math.isfinite(x) for x in r['losses']+r['grad_norms']+[r['initial_loss']]), 'Incomplete/nonfinite trajectory')
        if mode != 'profile':
            require(r['timed_steps'] == count and math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds'] > 0, 'Truncated/invalid timing')
        routes = module_routes(r['observed_shapes'], policy)
        require(r['module_policies'] == r['post_audit_routes'] == routes, 'Scope changed after stop_audit')
        names = set(routes); selected = {n for n, route in routes.items() if route != 'native_bf16'}
        require(set(r['target_modules']) == names and r['warmup_calls'] == {n: warmup*4 for n in names} and
                r['timed_calls'] == {n: count*4 for n in names}, 'Missing/incorrect actual projection calls')
        require(r['enabled_modules'] == sorted(selected) and r['replaced_modules'] == r['independent_scope_instances'] == len(selected),
                'Wrong scope count/shared module instance')
        require(r['forward_preserved'] == {n: route not in ('candidate', 'flux_tensor') for n, route in routes.items()}, 'Native forward replaced')
        placements = {'native_bf16': 'native', 'tensor': 'tensor', 'candidate': 'fused', 'flux_tensor': 'tensor'}
        require(r['codec_placements'] == {n: placements[route] for n, route in routes.items()}, 'Wrong codec placement')
        if is_native(policy):
            require(r['flux_library_loaded'] is False and r['native_linear_forward_preserved'] is True and
                    not any(k.startswith('libflux') for k in r), 'Native baseline loaded Flux')
            require(not r['workspace_checks'], 'Native scope registered candidate workspaces')
        else:
            require(len(v['libraries']) == 2 and all(r[name] == val for name, val in v['libraries'].items()), 'Loaded binary hash/path mismatch')
            require(set(r['workspace_checks']) == selected, 'Missing candidate workspace identities')
            for name, slots in r['workspace_checks'].items():
                role = module_role(name)
                if flux_tensor:
                    require(len(slots) == 1 and set(slots[0]) == {'shape', 'gemm_output_ptr'} and
                            slots[0]['shape'] == SHAPES[role], 'Wrong GemmOnly output workspace')
                elif fused_v2:
                    require(len(slots) == 1 and set(slots[0]) == {'shape', 'packet_ptr'} and
                            slots[0]['shape'] == SHAPES[role], 'Wrong original fused-v2 workspace')
                else:
                    require(len(slots) == 2 and all(s['shape'] == SHAPES[role] and
                            s['mask_sha256'] == role_records[role]['mask_sha256'] for s in slots), 'Workspace shape/mask crossed roles')
            for field in (('gemm_output_ptr',) if flux_tensor else ('packet_ptr',) if fused_v2 else ('packet_ptr', 'mask_ptr')):
                pointers = [s[field] for slots in r['workspace_checks'].values() for s in slots]
                require(all(type(p) is int and p > 0 for p in pointers) and len(pointers) == len(set(pointers)), 'Workspaces alias across modules/slots')
        require(set(r['selection_checks']) == selected, 'Wrong quantized module set')
        for name, check in r['selection_checks'].items():
            role = module_role(name); record = role_records[role]; mask = record['entry']['mask']
            expected = dict(role=role, shape=SHAPES[role], bf16_wire_bytes=7*256*4096*2,
                            mask_sha256=None if is_native(policy) or fused_v2 or flux_tensor else mask_sha(mask),
                            plan_sha256=None if is_native(policy) or fused_v2 or flux_tensor else record['entry_sha256'],
                            wire_bytes=7*256*32*136 if is_native(policy) or fused_v2 or flux_tensor else 7*256*4096*2+sum(mask[rank])*128*(136-256))
            require(check == expected, 'Wrong role/full-shape physical mask or wire bytes')
        if mode == 'smoke':
            require(set(r['forward_checks']) == selected, 'Missing/extra enabled-module numerical checks')
            for check in r['forward_checks'].values():
                require(check['passed'] and check['budget_vs_bf16'] == .05 and check['budget_vs_codec'] == .01 and
                        all(math.isfinite(check[k]) for k in ('max_abs', 'relative_l2', 'protocol_relative_l2')) and
                        check['relative_l2'] <= .05 and check['protocol_relative_l2'] <= .01, 'Numerical budget failed')
    return rows


def profile_roles(events, policy, routing=scope_protocol):
    """Associate CUDA kernels with role scopes through their runtime launch correlation."""
    SHAPES, definition, is_native = routing.SHAPES, routing.definition, routing.is_native
    flux_tensor = getattr(routing, 'is_flux_tensor', lambda p: False)(policy)
    result = {}
    for role in SHAPES:
        route = ('flux_tensor' if flux_tensor else 'tensor' if is_native(policy) else 'candidate') if role in definition(policy)['roles'] else 'native_bf16'
        name = f'scope.{role}.{route}'
        # CUDA launch correlations belong to CPU spans; GPU annotations mirror them.
        scopes = [e for e in events if e.get('name') == name and e.get('ph') == 'X'
                  and e.get('cat') == 'user_annotation']
        require(len(scopes) == 32*4, f'Missing profile scopes: {name}')
        by_thread = {}
        for s in scopes:
            by_thread.setdefault((s.get('pid'), s.get('tid')), []).append(s)
        intervals = {}
        for thread, spans in by_thread.items():
            spans.sort(key=lambda s: s['ts'])
            intervals[thread] = ([s['ts'] for s in spans], spans)
        correlations = set()
        for e in events:
            thread = (e.get('pid'), e.get('tid'))
            if e.get('cat') not in ('cuda_runtime', 'cuda_driver') or thread not in intervals:
                continue
            starts, spans = intervals[thread]; timestamp = e.get('ts', -1)
            i = bisect_right(starts, timestamp)-1
            if i >= 0 and timestamp <= spans[i]['ts']+spans[i]['dur']:
                correlations.add(e.get('args', {}).get('correlation'))
        correlations.discard(None)
        names = sorted({e['name'] for e in events if e.get('cat') == 'kernel' and
                        e.get('args', {}).get('correlation') in correlations})
        require(names, f'No correlated GPU kernels for {name}')
        if route == 'flux_tensor':
            require(any(('cutlass' in n.lower() or 'flux_bf16' in n.lower()) and 'sm90' in n.lower() for n in names),
                    'Native SM90 Flux GemmOnly kernel missing')
            require(not any('taco_decode_ring_kernel' in n or 'reducescatter' in n.lower().replace('_', '') for n in names),
                    'Naive Flux must not run fused codec or an extra ReduceScatter')
        elif route == 'candidate':
            require(any('flux_bf16' in n and 'sm80' in n.lower() for n in names) and
                    any('taco_decode_ring_kernel' in n for n in names), f'V2/mixed-wire kernel missing for {role}')
            require(not any('flux_bf16' in n and 'sm90' in n.lower() for n in names), 'Selected projection fell back to V3')
        else:
            require(not any('flux_bf16' in n or 'taco_decode_ring_kernel' in n for n in names), 'Native projection executed a candidate kernel')
        result[role] = dict(route=route, calls=len(scopes), kernels=names)
    if is_native(policy) or flux_tensor:
        native_profile(events)
    if is_native(policy):
        require(not any(e.get('cat') == 'kernel' and ('flux_bf16' in e['name'] or 'taco_decode_ring_kernel' in e['name'])
                        for e in events), 'Native scope executed Flux')
    return result


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
            rows = window(root, folder, policy, mode)
            key = fingerprints(rows)
            if reference is None: reference = key
            require(key == reference, 'Cross-scope initialization/RNG/tokens/software mismatch')
            files += [p for p in folder.rglob('*') if p.is_file()]
        for rank in range(8):
            with gzip.open(root/'results'/('profile-'+policy)/f'rank{rank}-profile.json.gz', 'rt') as f:
                profiles[f'{policy}/rank{rank}'] = profile_roles(json.load(f)['traceEvents'], policy)
        if not is_native(policy):
            for role in definition(policy)['roles']:
                path = root/'results'/f'mapping-{policy}-{role}.csv'
                with path.open() as f:
                    rows = [{k: int(v) for k, v in r.items()} for r in csv.DictReader(f)]
                require(len(rows) == 4096 and {r['rank'] for r in rows} == set(range(8)), 'Incomplete GPU mapping')
                for rank in range(8):
                    rr = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
                    require([r['index'] for r in rr] == list(range(512)) and
                            [r['m']*32+r['n'] for r in rr] == plans['roles'][role]['entry']['coords'][rank] and
                            all(r['destination'] == r['m']//2 and r['sm_occupancy'] > 0 for r in rr), 'Wrong role GPU mapping')
                files.append(path)
    gate = dict(protocol=PROTOCOL, completed=True, submission_sha256=sha(root/'submission.json'),
                fingerprints=reference, role_profiles=profiles, files={str(p.relative_to(root)): sha(p) for p in files})
    write(root/'preflight.json', gate)
    return gate


def summarize(values, blocks):
    require(set(values) == set(POLICIES) and all(len(v) == len(blocks) for v in values.values()), 'Incomplete paired blocks')
    require(all(math.isfinite(x) and x > 0 for v in values.values() for x in v), 'Invalid timings')
    if not blocks:
        return dict(status='无可用配对块', blocks=0, comparisons={}, speedups={})
    comparisons = {}
    for a, b in COMPARISONS:
        logs = [math.log(x/y) for x, y in zip(values[a], values[b])]
        stats = paired(logs) if len(blocks) > 1 else dict(percent=100*(1-math.exp(logs[0])), ci95=None, blocks=1)
        comparisons[a+'/'+b] = dict(stats, log_ratios=logs)
    speedups = {}
    for a, b in PRIMARY:
        ratios = [y/x for x, y in zip(values[a], values[b])]; best = max(range(len(ratios)), key=ratios.__getitem__)
        speedups[a] = dict(reference=b, median=statistics.median(ratios), ratios=ratios,
                          best=dict(ratio=ratios[best], **blocks[best], reference_ms=values[b][best], candidate_ms=values[a][best]))
    return dict(status='ok', blocks=len(blocks), block_ids=blocks, ms_per_step=values, comparisons=comparisons, speedups=speedups,
                median_ms={k: statistics.median(v) for k, v in values.items()},
                median_tokens_per_second={k: statistics.median(8192000/x for x in v) for k, v in values.items()})


def sensitivity(values, blocks):
    medians = {k: statistics.median(v) for k, v in values.items()}; keep = []; excluded = []
    for i, block in enumerate(blocks):
        triggers = {k: dict(ms=values[k][i], threshold_ms=1.1*medians[k]) for k in POLICIES if values[k][i] > 1.1*medians[k]}
        if triggers: excluded.append(dict(block, triggers=triggers))
        else: keep.append(i)
    return dict(original_median_ms=medians, excluded=excluded, remaining_blocks=len(keep),
                summary=summarize({k: [v[i] for i in keep] for k, v in values.items()}, [blocks[i] for i in keep]))


def markdown(result):
    lines = ['# H100 GPT6.7B TP8 Attention/MLP四组作用范围对比', '',
             '全样本为主；同作用范围配对。每窗口取8rank最大elapsed/20，8192 token/step。',
             '双模块相对Attention-only是扩展整套实现的净变化，不是单独增加量化的纯算法收益。',
             '最好加速比是最佳单块观测；95%区间按完整配对块bootstrap 10000次，seed20261009。无4%验收门槛。', '']
    for label, rounds in [('全样本', result['rounds']), ('慢窗口敏感性附表', {k: v['summary'] for k, v in result['sensitivity'].items()})]:
        lines += [f'## {label}', '', '| 轮次 | 策略 | 中位ms/step | tokens/s |', '| --- | --- | ---: | ---: |']
        for rep, s in rounds.items():
            if not s['blocks']: lines.append(f'| {rep} | 无可用配对块 | — | — |')
            for k, ms in s.get('median_ms', {}).items():
                lines.append(f'| {rep} | {k} | {ms:.3f} | {s["median_tokens_per_second"][k]:.2f} |')
        lines += ['', '| 轮次 | 候选 | 同范围基线 | 配对中位加速比 |', '| --- | --- | --- | ---: |']
        for rep, s in rounds.items():
            for k, v in s['speedups'].items(): lines.append(f'| {rep} | {k} | {v["reference"]} | {v["median"]:.6f} |')
        lines += ['', '| 轮次 | 候选 | 最好加速比 | 所在轮/block(0起) | 基线ms | 候选ms |', '| --- | --- | ---: | --- | ---: | ---: |']
        for rep, s in rounds.items():
            for k, v in s['speedups'].items():
                b = v['best']; lines.append(f'| {rep} | {k} | {b["ratio"]:.6f} | {b["round"]}/{b["block"]} | {b["reference_ms"]:.3f} | {b["candidate_ms"]:.3f} |')
        lines += ['', '| 轮次 | 比较 | 配对耗时下降% | 95%区间 |', '| --- | --- | ---: | --- |']
        for rep, s in rounds.items():
            for k, v in s['comparisons'].items():
                ci = v['ci95']; text = f'[{ci[0]:.3f}, {ci[1]:.3f}]' if ci else '不足两块，不估计区间'
                lines.append(f'| {rep} | {k} | {v["percent"]:.3f} | {text} |')
    lines += ['', '正确性及完整性通过；性能收益按区间单独判断，负值和跨零区间均保留。']
    for rep, s in result['sensitivity'].items():
        lines.append(f'筛选轮次 {rep}：剩余{s["remaining_blocks"]}块；剔除及触发策略：`{s["excluded"]}`。')
    return '\n'.join(lines)+'\n'


def report(root, offline=False):
    root = Path(root); info = verify_inputs(root, offline); gate = read(root/'preflight.json')
    require(info['executable'] and not info['source_only'] and gate['completed'] and gate['protocol'] == PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json'), 'Completed GPU preflight required')
    verify_files(root, gate['files'])
    result = dict(protocol=PROTOCOL, run_id=info['run_id'], completed=False, offline=offline, all_samples=[], rounds={}, sensitivity={},
                  acceptance_threshold=None, sample_policy='all samples primary', correctness_passed=True,
                  bootstrap=dict(resamples=10000, seed=20261009, unit='complete paired block'),
                  config=info['config'], candidate_source_commit=info['candidate_source_commit'],
                  measurement_identity=measurement_identity(info, gate),
                  role_plan_sha256=read(root/'scripts/config.json')['role_plan_sha256'])
    combined = {k: [] for k in POLICIES}; combined_blocks = []; files = {}
    for rep in (1, 2):
        values = {k: [] for k in POLICIES}; blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == 4 and {j['policy'] for j in jobs} == set(POLICIES), 'Incomplete block')
            for job in jobs:
                folder = root/'results'/job['directory']
                rows = window(root, folder, job['policy'], 'timing', job['block'], job['index'])
                require(fingerprints(rows) == [tuple(x) for x in gate['fingerprints']], 'Timing initialization/RNG/tokens/software changed')
                elapsed = [r['elapsed_seconds'] for r in rows]; ms = max(elapsed)*1000/20
                values[job['policy']].append(ms)
                result['all_samples'].append(dict(job, rank_elapsed_seconds=elapsed, ms_per_step=ms, tokens_per_second=8192000/ms))
                files.update({str(p.relative_to(root)): sha(p) for p in folder.rglob('*') if p.is_file()})
        result['rounds'][str(rep)] = summarize(values, blocks); result['sensitivity'][str(rep)] = sensitivity(values, blocks)
        for k in POLICIES: combined[k] += values[k]
        combined_blocks += blocks
    result['rounds']['combined'] = summarize(combined, combined_blocks)
    require(len(result['all_samples']) == info['timing_windows'], 'Missing timing windows')
    result.update(completed=True, timing_windows=info['timing_windows'], timing_rank_records=8*info['timing_windows'],
                  preflight_rank_records=64, profile_traces=32, candidate_shape_mappings=3,
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=files))
    write(root/'report.json', result); (root/'report.md').write_text(markdown(result))
    return result
