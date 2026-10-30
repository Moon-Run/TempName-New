"""Strict three-arm identity, execution-path, numerical and paired timing evidence."""
from bisect import bisect_right
import csv
import gzip
import json
import math
from pathlib import Path
import statistics

import a_baseline_build as fused_build
import a_baseline_protocol as routing
import a_baseline_training_sources as sources
import ablation3_report
import ablation4_runtime as communication
import ablation6_build as flux_build
import baseline_codec_build as codec_build
import baseline_codec_validation
import scope_report
from a_baseline_prepare import SHARED_ENVIRONMENT, model, runtime_root, shape_metadata
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
            'Wrong three-baseline protocol/model/scope')
    require(type(info['source_only']) is bool and info['executable'] == (not info['source_only']),
            'Rehearsal identity changed')
    verify_files(root, info['files'])
    expected = routing.schedule(info['blocks_per_round'], info['schedule_seed'], info['rounds'])
    require(info['schedule'] == read(root/'schedule.json') == expected and
            info['timing_windows'] == len(expected), 'Incomplete or altered three-arm schedule')
    cfg = read(root/'scripts/config.json')
    seal = read(root/'provenance/preparation.json')
    require(all(seal[k] == info[k] for k in seal), 'Preparation identity changed')
    require(info['execution_timing'] == routing.TIMING and all(type(cfg[k]) is int and cfg[k] == v for k, v in routing.TIMING.items()) and
            cfg['smoke_warmup_steps'] == 2 and cfg['smoke_steps'] == 3, 'Timing/smoke steps changed')
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
    require(sha(root/'scripts/routing.py') == sha(HERE/'a_baseline_protocol.py') and
            sha(root/'scripts/reference.py') == info['codec_identity']['reference_sha256'], 'Routing/reference changed')
    snapshots = {'build_manifest': 'fused-manifest.json', 'fused_identity': 'fused-identity.json',
                 'flux_build_manifest': 'gemm-only-manifest.json', 'codec_build_manifest': 'codec-manifest.json'}
    for key, filename in snapshots.items():
        require(info[key]['sha256'] == sha(root/'provenance'/filename), 'Frozen build snapshot changed: '+key)
    fused = read(root/'provenance/fused-manifest.json')
    ident = read(root/'provenance/fused-identity.json')
    gemm = read(root/'provenance/gemm-only-manifest.json')
    codec = read(root/'provenance/codec-manifest.json')
    flux_build.validate_snapshot(gemm, built=info['executable'])
    codec_build.validate_snapshot(codec, built=info['executable'])
    require(ident['protocol'] == fused_build.PROTOCOL and ident['build_manifest_sha256'] == info['build_manifest']['sha256'] and
            set(fused['variants']) == {'basic', 'improved'} and
            all(fused['variants'][p]['source_sha256'] == ident['variants'][p] for p in ('basic', 'improved')),
            'Frozen fused build identity changed')
    require(fused['source_only'] == ident['source_only'] == gemm['source_only'] == codec['source_only'] == info['source_only'] and
            info['codec_library'] == cfg['codec_library'] == codec['library'], 'Build/codec rehearsal mode changed')
    roots = (Path(info['build_manifest']['path']).parent, Path(info['flux_build_manifest']['path']).parent)
    require(set(info['variants']) == set(POLICIES), 'Missing/extra arms')
    for p in POLICIES:
        v = info['variants'][p]
        require(all(v.get(k) == value for k, value in routing.definition(p).items()) and v['sampling'] is False and
                Path(v['root']) == runtime_root(*roots, p), 'Policy implementation/runtime root changed')
        if info['executable']:
            require(set(v['libraries']) == {'libflux_cuda.so', 'libflux_cuda_ths_op.so'}, 'Missing runtime libraries')
            manifest = gemm if p == 'cuda' else fused
            prefix = flux_build.VARIANT if p == 'cuda' else p
            for name, identity in v['libraries'].items():
                require(identity['sha256'] == manifest['files'][prefix+'/'+name], 'Library differs from build manifest')
                if not offline:
                    require(Path(identity['path']) == (Path(v['root'])/name).resolve() and sha(identity['path']) == identity['sha256'],
                            'Loaded runtime binary changed')
        else:
            require(v['libraries'] == {}, 'CPU rehearsal cannot register executable libraries')
    if not offline:
        verify_files('/', info['controllers'])
        for key in snapshots:
            require(sha(info[key]['path']) == info[key]['sha256'], 'Live build manifest changed')
        fused_build.verify(roots[0], built=info['executable'])
        flux_build.verify(roots[1], built=info['executable'])
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
        expected = {n: (row['warmup_steps']+len(row['losses']))*4 for n in row['enabled_modules']} if policy == 'cuda' else {}
        require(row['codec_calls'] == expected and
                row['codec_library'] == (info['codec_library'] if policy == 'cuda' else None),
                'Standalone CUDA codec route/call count changed')
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
            if policy == 'cuda':
                baseline_codec_validation.verify(row['codec_validation'], info['codec_library'])
            else:
                require(row['codec_validation'] is None, 'Fused path must not call standalone codec prechecks')
        else:
            require(row['codec_validation'] is None and not row['smoke_output_sha256'] and not row['retained_output_checks'],
                    'Smoke diagnostics leaked into timing/profile')
    return rows


def equivalent_pair(basic, improved, smoke=False):
    require(len(basic) == len(improved) == 8, 'Eight ranks are required for equivalence')
    for a, b in zip(basic, improved):
        for field in ('losses', 'grad_norms', 'initial_loss', 'warmup_last_loss'):
            require(a[field] == b[field], 'Basic/improved numerical trajectory differs: '+field)
        if smoke:
            require(a['smoke_output_sha256'] == b['smoke_output_sha256'],
                    'Basic/improved model projection outputs are not bitwise equal')
    return True


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
        route = 'flux_tensor' if policy == 'cuda' else 'candidate'
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
            if policy == 'cuda':
                gemms = [e for e in path if ('cutlass' in e['name'].lower() or 'flux_bf16' in e['name'].lower()) and
                         'sm90' in e['name'].lower()]
                enc = [e for e in path if 'fused_ash_compress_kernel' in e['name']]
                dec = [e for e in path if 'fused_ash_decompress_kernel' in e['name']]
                require(len(gemms) == len(enc) == len(dec) == 1 and not barriers and not clones and
                        not any('taco_decode_ring_kernel' in n or 'taco_basic_' in n or
                                'reducescatter' in n.lower().replace('_', '') for n in names),
                        'CUDA baseline must use only GemmOnly and standalone codec, without fused RS/clone')
                require(any('nccl' in n.lower() for n in names), 'Standalone exchange lacks actual NCCL kernels')
                markers = [e for e in events if e.get('cat') == 'user_annotation' and contained(e, scope)]
                for name in ('ablation6.flux_gemm_only', 'megatron_taco.encode', 'megatron_taco.exchange', 'megatron_taco.decode_reduce'):
                    require(sum(e.get('name') == name for e in markers) == 1, 'Missing standalone stage: '+name)
                require(any('alltoall' in e.get('name', '').lower().replace('_', '') and contained(e, scope) for e in cpu_ops),
                        'Standalone NCCL All-to-All call missing')
                values.append(dict(gemm_us=gemms[0]['dur'], decode_us=dec[0]['dur'], encode_us=enc[0]['dur']))
            else:
                gemms = [e for e in path if 'flux_bf16' in e['name'] and 'sm80' in e['name'].lower() and 'reducescatter' in e['name']]
                names_expected = ['taco_basic_decode_kernel', 'taco_basic_reduce_kernel'] if policy == 'basic' else ['taco_decode_ring_kernel']
                stages = [[e for e in path if name in e['name']] for name in names_expected]
                require(len(gemms) == 1 and len(barriers) == 2 and len(clones) == 1 and all(len(x) == 1 for x in stages),
                        'Wrong basic/improved GEMM, receiver stages, barriers or output clone')
                allowed = set(names_expected)
                require(not any(('taco_' in n and ('decode' in n or 'reduce' in n) and not any(x in n for x in allowed)) or
                                'fused_ash_' in n or 'taco_ma' in n or 'taco_op' in n for n in names),
                        'Another decoder/optimization replaced the declared path')
                ordered = [gemms[0], barriers[0], *[x[0] for x in stages], barriers[1]]
                require(all(e.get('args', {}).get('stream') is not None and math.isfinite(e['dur']) and e['dur'] >= 0 for e in ordered) and
                        len({e['args']['stream'] for e in ordered}) == 1 and
                        all(a['ts']+a['dur'] <= b['ts']+.01 for a, b in zip(ordered, ordered[1:])),
                        'Producer/publication/receiver/reuse stream order changed')
                values.append(dict(gemm_us=gemms[0]['dur'], decode_us=sum(x[0]['dur'] for x in stages)))
        result[role] = dict(calls=128, route=route, barriers=0 if policy == 'cuda' else 256,
                            output_clones=0 if policy == 'cuda' else 128,
                            receiver_kernel_calls=256 if policy == 'basic' else 128,
                            gemm_median_us=statistics.median(x['gemm_us'] for x in values),
                            receiver_median_us=statistics.median(x['decode_us'] for x in values))
    return result


def verify_mapping(rows):
    require(len(rows) == 4096 and {r['rank'] for r in rows} == set(range(8)), 'Incomplete original-order mapping')
    for rank in range(8):
        rr = sorted((r for r in rows if r['rank'] == rank), key=lambda r: r['index'])
        require([r['index'] for r in rr] == list(range(512)) and
                sorted(r['m']*32+r['n'] for r in rr) == list(range(512)) and
                sorted(r['base_m']*32+r['base_n'] for r in rr) == list(range(512)) and
                all(r['destination'] == r['m']//2 and r['sm_occupancy'] > 0 and
                    r['m'] == (r['base_m']+2*rank)%16 and r['n'] == r['base_n'] for r in rr),
                'Original Flux order was replaced or lost tiles')
    return [{k: r[k] for k in ('rank', 'index', 'm', 'n', 'destination', 'base_m', 'base_n')} for r in rows]


def preflight(root, offline=False):
    root = Path(root)
    info = verify_inputs(root, offline)
    require(info['executable'] and not info['source_only'], 'CPU rehearsals cannot certify GPU execution')
    start = read(root/'start-gpu-check.json')
    require(start['passed'] is True and start['run_id'] == info['run_id'], 'Missing fresh startup check')
    files, reference, profiles, records, mappings = [root/'start-gpu-check.json'], None, {}, {}, {}
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
        if routing.needs_mapping(p):
            for role in routing.SHAPES:
                path = root/'results'/f'mapping-{p}-{role}.csv'
                with path.open() as handle:
                    mappings[(p, role)] = verify_mapping([{k: int(v) for k, v in row.items()} for row in csv.DictReader(handle)])
                files.append(path)
    for mode in ('smoke', 'profile'):
        equivalent_pair(records[('basic', mode)], records[('improved', mode)], smoke=mode == 'smoke')
    require(all(mappings[('basic', r)] == mappings[('improved', r)] for r in routing.SHAPES), 'Fusion-pair mapping changed')
    gate = dict(protocol=PROTOCOL, completed=True, correctness_passed=True, execution_path_validated=True,
                bitwise_equal_within_pairs=True, equivalence_pairs=routing.EQUIVALENCE_PAIRS,
                submission_sha256=sha(root/'submission.json'), communication=info['communication'],
                codec_identity=info['codec_identity'], codec_library=info['codec_library'], fingerprints=reference,
                role_profiles=profiles, files={str(p.relative_to(root)): sha(p) for p in files})
    write(root/'preflight.json', gate)
    return gate


def summarize(values, blocks, samples):
    result = ablation3_report.summarize(values, blocks, POLICIES, routing.COMPARISONS)
    result['mean_ms'] = {p: statistics.mean(v) for p, v in values.items()}
    result['tokens_per_second'] = {p: 8192000/ms for p, ms in result['mean_ms'].items()}
    result['mean_loss'] = {p: statistics.mean(s['mean_loss'] for s in samples if s['policy'] == p) for p in POLICIES}
    result['mean_final_loss'] = {p: statistics.mean(s['final_loss'] for s in samples if s['policy'] == p) for p in POLICIES}
    return result


def markdown(result):
    lines = ['# Flux 基础融合、改进融合与独立 CUDA 量化：同场完整训练', '',
             f'本次 {result["timing_windows"]} 个正式窗口、{result["formal_optimizer_steps"]} 个正式 optimizer step；每窗口 10 步预热＋50 步计时。',
             '三组均覆盖 Attention＋MLP、全远端 H128/E4M3；没有新排序、预取、Graph 或提交缓存。',
             'basic/improved 保持相同原生产顺序与数值语义；CUDA 使用不同 GEMM/codec 后端，单独检查误差，不要求跨后端逐位一致。',
             '期望耗时 improved < basic < cuda 仅是待检验假设；负结果全部保留，不据此中止后续比较。']
    for rep, row in result['rounds'].items():
        lines += ['', f'## {rep}', '', '| 指标 | '+' | '.join(routing.LABELS[p] for p in POLICIES)+' |',
                  '| --- | ---: | ---: | ---: |']
        for field, label, fmt in [('mean_ms', '平均 ms/step', '.3f'), ('median_ms', '中位 ms/step', '.3f'),
                                  ('mean_loss', '平均 loss', '.6f'), ('mean_final_loss', '末步 loss 平均', '.6e')]:
            lines.append('| '+label+' | '+' | '.join(format(row[field][p], fmt) for p in POLICIES)+' |')
        lines += ['', '| 比较 | 配对耗时下降 %［95% CI］ |', '| --- | ---: |']
        for key, stat in row['comparisons'].items():
            lines.append(f'| {key} | {stat["percent"]:+.3f} [{stat["ci95"][0]:+.3f}, {stat["ci95"][1]:+.3f}] |')
    lines += ['', f'期望顺序点估计成立：{result["expected_order_point_estimate"]}；两项相邻比较区间均为正：{result["expected_order_confirmed"]}。',
              '主结果保留全部完整块；按配对块 bootstrap 10000 次，未做多重比较校正。',
              '独立窗口累计训练步数不等于一条连续训练轨迹；固定合成数据不代替真实语料质量验证。']
    return '\n'.join(lines)+'\n'


def report(root, offline=False, save=True):
    root = Path(root)
    info = verify_inputs(root, offline)
    gate = read(root/'preflight.json')
    require(info['executable'] and not info['source_only'] and gate['completed'] and gate['correctness_passed'] and
            gate['execution_path_validated'] and gate['bitwise_equal_within_pairs'] and
            gate['equivalence_pairs'] == routing.EQUIVALENCE_PAIRS and gate['protocol'] == PROTOCOL and
            gate['submission_sha256'] == sha(root/'submission.json') and
            all(gate[k] == info[k] for k in ('communication', 'codec_identity', 'codec_library')),
            'Fresh complete three-arm preflight is required')
    verify_files(root, gate['files'])
    require({str(p.relative_to(root/'results')) for p in (root/'results').glob('round*/window-*') if p.is_dir()} ==
            {j['directory'] for j in info['schedule']}, 'Missing/extra formal windows')
    result = dict(protocol=PROTOCOL, run_id=info['run_id'], completed=False, correctness_passed=True,
                  execution_path_validated=True, offline=offline, comparison_scope='both',
                  bitwise_equal_within_pairs=True, identical_trajectories_within_pairs=True,
                  equivalence_pairs=routing.EQUIVALENCE_PAIRS, full_optimizer_steps_measured=True,
                  round_count=2, blocks_per_round=info['blocks_per_round'], config=info['config'],
                  execution_timing=dict(routing.TIMING), communication=info['communication'],
                  codec_identity=info['codec_identity'], codec_library=info['codec_library'],
                  measurement_identity=measurement_identity(info, gate), rounds={}, all_samples=[],
                  sample_policy='all complete blocks; no post-hoc exclusions', acceptance_threshold=None)
    combined, all_blocks, files = {p: [] for p in POLICIES}, [], {}
    for rep in (1, 2):
        values, samples = {p: [] for p in POLICIES}, []
        blocks = [dict(round=rep, block=b) for b in range(info['blocks_per_round'])]
        for block in blocks:
            jobs = [j for j in info['schedule'] if j['round'] == rep and j['block'] == block['block']]
            require(len(jobs) == 3 and {j['policy'] for j in jobs} == set(POLICIES), 'Incomplete three-arm pairing block')
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
            equivalent_pair(paired_rows['basic'], paired_rows['improved'])
        result['rounds'][str(rep)] = summarize(values, blocks, samples)
        for p in POLICIES:
            combined[p] += values[p]
        all_blocks += blocks
        result['all_samples'] += samples
    result['rounds']['combined'] = summarize(combined, all_blocks, result['all_samples'])
    estimates = result['rounds']['combined']['comparisons']
    adjacent = [estimates[k] for k in ('improved/basic', 'basic/cuda')]
    result.update(completed=True, timing_windows=info['timing_windows'],
                  formal_optimizer_steps=routing.TIMING['timed_steps']*info['timing_windows'],
                  timing_rank_records=8*info['timing_windows'], preflight_rank_records=48,
                  profile_traces=24, candidate_shape_mappings=4,
                  expected_order_point_estimate=all(s['percent'] > 0 for s in adjacent),
                  expected_order_confirmed=all(s['percent'] > 0 and s['ci95'][0] > 0 for s in adjacent),
                  evidence=dict(preflight_sha256=sha(root/'preflight.json'), timing_files=files))
    if save:
        write(root/'report.json', result)
        (root/'report.md').write_text(markdown(result))
    return result
