"""Strict 50-step evidence, shared CUDA-codec gates and five-policy reports."""
import csv
import gzip
import json
from pathlib import Path
import statistics

import ablation3_report
import ablation4_build as parent_build
import ablation4_report
import ablation4_runtime as communication
import ablation6_build as flux_build
import baseline_codec_build as codec_build
import baseline_codec_validation
import baseline_protocol as routing
import baseline_sources as sources
from baseline_prepare import role_hashes, runtime_root
from ablation_report import measurement_identity
from common import read, require, sha, verify_files, write
from report import fingerprints
from scope_plan import model_config, hardware_identity
import scope_report

PROTOCOL, POLICIES = routing.PROTOCOL, routing.POLICIES


def verify_inputs(root, offline=False):
    root=Path(root); info=read(root/'submission.json'); verify_files(root,info['files'])
    require(info['protocol']==PROTOCOL and info['config']==model_config(),'Wrong baseline protocol/model')
    expected=routing.schedule(info['blocks_per_round'],info['schedule_seed'],info['rounds'])
    require(info['schedule']==read(root/'schedule.json')==expected and info['timing_windows']==len(expected),'Five-policy schedule changed')
    cfg=read(root/'scripts/config.json'); seal=read(root/'provenance/preparation.json')
    require(all(seal[k]==info[k] for k in ('protocol','run_id','source_only','executable','rounds','communication','execution_timing','codec_identity','codec_library')) and
            info['executable']==(not info['source_only']),'Preparation identity changed')
    require(info['execution_timing']==routing.TIMING and all(type(cfg[k]) is int and cfg[k]==v for k,v in routing.TIMING.items()),
            'TACO CUDA baseline requires 10 warmup / 50 timed steps')
    require(info['codec_identity']==cfg['codec_identity']==sources.codec_identity() and
            sha(root/'scripts/megatron_taco.py')==info['codec_identity']['source_sha256'] and
            sha(root/'scripts/reference.py')==info['codec_identity']['reference_sha256'],'Shared CUDA codec changed')
    require(cfg['protocol']==PROTOCOL and cfg['policies']==list(POLICIES) and cfg['repetitions']==info['rounds'] and
            cfg['communication']==communication.validate(info['communication']) and cfg['graph_forward'] is False and
            cfg['double_buffered'] is True and cfg['ring_reduction'] is True,'Worker execution path changed')
    model=model_config()
    require(cfg['common']=={k:model[k] for k in ('layers','micro_batch','global_batch','seed','token_seed','vocab')} and
            cfg['cases']==[dict(name='h100-phase6',**{k:model[k] for k in ('hidden','ffn','heads','sequence','tp')})],
            'Worker model/shapes changed')
    plans=read(root/'scripts/selection-plan.json'); parent=read(root/'provenance/build-manifest.json')
    candidate=read(root/'provenance/candidate-manifest.json'); fused=read(root/'provenance/fused-manifest.json')
    fm=read(root/'provenance/gemm-only-manifest.json')
    cm=read(root/'provenance/codec-manifest.json')
    codec_build.validate_snapshot(cm,built=info['executable'])
    require(sha(root/'provenance/codec-manifest.json')==info['codec_build_manifest']['sha256'] and
            info['codec_library']==cfg['codec_library']==cm['library'] and cm['source_only']==info['source_only'],
            'Standalone CUDA library identity changed')
    require((root/'scripts/megatron_taco.py').read_text()==sources.codec_source() and
            (root/'scripts/worker.py').read_text()==sources.worker_source(info['source_only']) and
            (root/'scripts/adapter.py').read_text()==sources.adapter_source() and
            (root/'scripts/taco_support.py').read_text()==sources.taco_source(), 'Worker/adapter execution source changed')
    require(sha(root/'scripts/selection-plan.json')==seal['plans_sha256']==parent['files']['plans.json'] and
            cfg['role_plan_sha256']==role_hashes(plans),'A/B/C plans changed')
    require(sha(root/'provenance/build-manifest.json')==info['build_manifest']['sha256'] and
            sha(root/'provenance/candidate-manifest.json')==parent['candidate_manifest_sha256'] and
            sha(root/'provenance/gemm-only-manifest.json')==info['flux_build_manifest']['sha256'],'Build snapshot changed')
    parent_build.validate_snapshots(parent,plans,candidate,fused,info['executable'],verify_evidence=not offline)
    flux_build.validate_snapshot(fm,built=info['executable'])
    require(parent['source_only']==fm['source_only']==info['source_only'],'Build rehearsal identity changed')
    if info['executable']:
        require(all(hardware_identity(p['hardware'])==hardware_identity(info['calibration_hardware']) for p in plans['parent']['roles'].values()),
                'Calibration hardware/order changed')
    require(set(info['variants'])==set(POLICIES),'Missing/extra policies')
    build_root=Path(info['build_manifest']['path']).parent; flux_root=Path(info['flux_build_manifest']['path']).parent
    for p in POLICIES:
        v=info['variants'][p]
        require(all(v.get(k)==value for k,value in routing.definition(p).items()),'Implementation/quantization route changed')
        if routing.is_native(p):
            require(not any(k in v for k in ('root','libraries','sampling')),'Megatron must not load Flux'); continue
        require(Path(v['root'])==runtime_root(build_root,flux_root,p) and v['sampling'] is False,'Wrong build route')
        if info['executable']:
            require(set(v['libraries'])=={'libflux_cuda.so','libflux_cuda_ths_op.so'},'Runtime libraries missing')
            if not offline:
                for lib in v['libraries'].values(): require(sha(lib['path'])==lib['sha256'],'Runtime library changed')
    if not offline:
        verify_files('/',info['build_files'])
        for key in ('build_manifest','flux_build_manifest','codec_build_manifest'):
            require(sha(info[key]['path'])==info[key]['sha256'],'Build manifest changed')
    return info


def window(root, folder, policy, mode, block=-1, index=-1):
    root,folder=Path(root),Path(folder)
    rows=scope_report.window(root,folder,policy,mode,block,index,routing)
    info=read(root/'submission.json'); command=read(folder/'command.json')
    require(command['communication']==info['communication'] and command['scope']==routing.definition(policy) and
            all(command[k]==v for k,v in dict(policy=policy,mode=mode,block=block,index=index).items()),'Command identity changed')
    for row in rows:
        require(row['communication']==info['communication'] and row['losses']==rows[0]['losses'],'TP loss/communication mismatch')
        require(row['codec_identity']==info['codec_identity'],'Loaded CUDA codec differs from shared source')
        expected={name:(row['warmup_steps']+len(row['losses']))*4 for name in row['enabled_modules']} if routing.is_tensor(policy) else {}
        require(row['codec_calls']==expected,'Shared TensorTacoRS was not called for every projection/microbatch')
        require(row['codec_library']==(info['codec_library'] if routing.is_tensor(policy) else None),
                'Baseline library not executed, or A/B/C loaded the new codec')
        if routing.is_tensor(policy) and mode=='smoke':
            baseline_codec_validation.verify(row['codec_validation'],info['codec_library'])
        else:
            require(row['codec_validation'] is None,'Codec diagnostics must stay outside model timing/profile')
    return rows


def profile_roles(events, policy):
    result=scope_report.profile_roles(events,policy,routing)
    if routing.is_tensor(policy):
        for name in ('megatron_taco.encode','megatron_taco.exchange','megatron_taco.decode_reduce'):
            require(sum(e.get('name')==name and e.get('ph')=='X' and e.get('cat')=='user_annotation' for e in events)==256,
                    'Missing shared Tensor TACO profile calls: '+name)
        for kernel in ('fused_ash_compress_kernel', 'fused_ash_decompress_kernel'):
            require(sum(kernel in e.get('name','') and e.get('cat')=='kernel' for e in events)==256,
                    'Expected 256 actual upstream CUDA kernel calls: '+kernel)
        require(not any('taco_decode_ring_kernel' in e.get('name','') for e in events),
                'Standalone baseline must not execute fused-v2 decode/reduce')
    if routing.is_flux_tensor(policy):
        require(sum(e.get('name')=='ablation6.flux_gemm_only' and e.get('ph')=='X' and e.get('cat')=='user_annotation' for e in events)==256,
                'Flux baseline must execute 256 local GemmOnly calls')
    return result


def preflight(root, offline=False):
    root=Path(root); info=verify_inputs(root,offline)
    require(info['executable'] and not info['source_only'],'CPU rehearsal cannot certify GPU execution')
    start=read(root/'start-gpu-check.json')
    require(start['passed'] and start['run_id']==info['run_id'],'Missing startup check')
    files=[root/'start-gpu-check.json']; reference=None; profiles={}; plans=read(root/'scripts/selection-plan.json')
    for p in POLICIES:
        for mode in ('smoke','profile'):
            folder=root/'results'/f'{mode}-{p}'; rows=window(root,folder,p,mode); key=fingerprints(rows)
            if reference is None: reference=key
            require(key==reference,'Cross-policy initialization/data/environment changed')
            files += [f for f in folder.rglob('*') if f.is_file()]
        for rank in range(8):
            with gzip.open(root/'results'/f'profile-{p}'/f'rank{rank}-profile.json.gz','rt') as f:
                profiles[f'{p}/rank{rank}']=profile_roles(json.load(f)['traceEvents'],p)
        if routing.needs_mapping(p):
            for role in routing.SHAPES:
                path=root/'results'/f'mapping-{p}-{role}.csv'
                with path.open() as f:
                    ablation4_report.verify_mapping([{k:int(v) for k,v in row.items()} for row in csv.DictReader(f)],p,role,plans)
                files.append(path)
    gate=dict(protocol=PROTOCOL,completed=True,submission_sha256=sha(root/'submission.json'),communication=info['communication'],
              codec_identity=info['codec_identity'],codec_library=info['codec_library'],fingerprints=reference,role_profiles=profiles,
              files={str(p.relative_to(root)):sha(p) for p in files})
    write(root/'preflight.json',gate); return gate


def summarize(values, blocks, samples):
    result=ablation3_report.summarize(values,blocks,POLICIES,routing.COMPARISONS)
    if blocks:
        result['mean_ms']={p:statistics.mean(v) for p,v in values.items()}
        result['tokens_per_second']={p:8192000/ms for p,ms in result['mean_ms'].items()}
        result['mean_loss']={p:statistics.mean(s['mean_loss'] for s in samples if s['policy']==p) for p in POLICIES}
        result['mean_final_loss']={p:statistics.mean(s['final_loss'] for s in samples if s['policy']==p) for p in POLICIES}
    return result


def markdown(result):
    order=[POLICIES[4],POLICIES[2],POLICIES[3],POLICIES[1],POLICIES[0]]
    lines=['# H100 新baseline：消融6 A/B/C与两种独立TACO CUDA量化实现','',
        f'每窗口预热10步、正式50步；{result["timing_windows"]}个正式窗口，{result["formal_optimizer_steps"]}个正式optimizer step。',
        'Flux使用原生Hopper GemmOnly；Megatron保留原生GEMM。两者调用同一份作者独立CUDA编解码kernel，通信替换为NCCL All-to-All。A/B/C与消融6一致。',
        '这是原版TACO codec的移植基线，不包含COCCL通信调度，不代表论文完整系统；没有GEMM-量化或解码-归约融合。',
        '两个baseline的codec/通信一致，不要求不同GEMM产生逐位相同输出。C/B同时改变mask与候选实现；A/Flux同时改变GEMM、融合和调度。']
    for rep,s in result['rounds'].items():
        lines += ['',f'## {rep}','','| 指标 | '+' | '.join(routing.LABELS[p] for p in order)+' |','| --- | '+' | '.join(['---:']*5)+' |']
        for field,label,fmt in (('mean_ms','平均耗时（ms/step）','.3f'),('median_ms','中位耗时（ms/step）','.3f'),
                                ('mean_loss','平均训练loss','.6f'),('mean_final_loss','最终训练loss的平均','.6e')):
            lines.append('| '+label+' | '+' | '.join(format(s[field][p],fmt) for p in order)+' |')
        lines += ['','| X / 参考Y | 配对耗时下降% [95% CI] |','| --- | ---: |']
        for key,v in s['comparisons'].items():
            a,b=key.split('/'); interval=f'[{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]' if v['ci95'] else '不足两块'
            lines.append(f'| {routing.LABELS[a]} / {routing.LABELS[b]} | {v["percent"]:+.3f} {interval} |')
    lines += ['','主表保留全样本；完整五组块bootstrap10000次、seed20261009，区间未作多重比较校正，无4%门槛。',
              'loss只计rank0的50个正式step，最终loss跨窗口平均；固定合成微批重复训练不代表真实语料泛化。']
    for rep,s in result['sensitivity'].items():lines.append(f'轮次{rep}：保留{s["remaining_blocks"]}块，排除：`{s["excluded"]}`。')
    return '\n'.join(lines)+'\n'


def report(root, offline=False):
    root=Path(root); info=verify_inputs(root,offline); gate=read(root/'preflight.json')
    require(info['executable'] and not info['source_only'] and gate['completed'] and gate['protocol']==PROTOCOL and
            gate['submission_sha256']==sha(root/'submission.json') and gate['communication']==info['communication'] and
            gate['codec_identity']==info['codec_identity'] and gate['codec_library']==info['codec_library'],'Fresh complete preflight required')
    verify_files(root,gate['files'])
    require({str(p.relative_to(root/'results')) for p in (root/'results').glob('round*/window-*') if p.is_dir()}==
            {j['directory'] for j in info['schedule']},'Missing/extra timing windows')
    result=dict(protocol=PROTOCOL,run_id=info['run_id'],completed=False,correctness_passed=True,offline=offline,
        round_count=info['rounds'],blocks_per_round=info['blocks_per_round'],config=info['config'],execution_timing=routing.TIMING,
        communication=info['communication'],codec_identity=info['codec_identity'],codec_library=info['codec_library'],all_samples=[],rounds={},sensitivity={},
        candidate_source_commit=info['candidate_source_commit'],role_plan_sha256=read(root/'scripts/config.json')['role_plan_sha256'],
        primary_comparisons=routing.PRIMARY,comparisons=routing.COMPARISONS,acceptance_threshold=None,
        measurement_identity=measurement_identity(info,gate),
        bootstrap=dict(resamples=10000,seed=20261009,unit='complete five-policy block',multiple_comparison_correction=None))
    result['measurement_identity']['communication']=info['communication']
    result['measurement_identity']['codec_identity']=info['codec_identity']
    result['measurement_identity']['codec_library']=info['codec_library']
    combined={p:[] for p in POLICIES}; all_blocks=[]; files={}
    for rep in range(1,info['rounds']+1):
        values={p:[] for p in POLICIES}; blocks=[dict(round=rep,block=b) for b in range(info['blocks_per_round'])]; samples=[]
        for block in blocks:
            jobs=[j for j in info['schedule'] if j['round']==rep and j['block']==block['block']]
            require(len(jobs)==5 and {j['policy'] for j in jobs}==set(POLICIES),'Incomplete five-policy block')
            for job in jobs:
                folder=root/'results'/job['directory']; rows=window(root,folder,job['policy'],'timing',job['block'],job['index'])
                require(fingerprints(rows)==[tuple(x) for x in gate['fingerprints']],'Timing identity changed')
                elapsed=[r['elapsed_seconds'] for r in rows]; ms=max(elapsed)*1000/routing.TIMING['timed_steps']
                values[job['policy']].append(ms)
                samples.append(dict(job,rank_elapsed_seconds=elapsed,ms_per_step=ms,tokens_per_second=8192000/ms,
                                    mean_loss=statistics.mean(rows[0]['losses']),final_loss=rows[0]['losses'][-1]))
                files.update({str(p.relative_to(root)):sha(p) for p in folder.rglob('*') if p.is_file()})
        result['rounds'][str(rep)]=summarize(values,blocks,samples)
        result['sensitivity'][str(rep)]=ablation3_report.sensitivity(values,blocks,POLICIES,routing.COMPARISONS)
        for p in POLICIES:combined[p]+=values[p]
        all_blocks+=blocks;result['all_samples']+=samples
    result['rounds']['combined']=summarize(combined,all_blocks,result['all_samples'])
    require(len(result['all_samples'])==info['timing_windows'],'Missing timing samples')
    result.update(completed=True,timing_windows=info['timing_windows'],formal_optimizer_steps=50*info['timing_windows'],
        timing_rank_records=8*info['timing_windows'],preflight_rank_records=80,profile_traces=40,candidate_shape_mappings=6,
        evidence=dict(preflight_sha256=sha(root/'preflight.json'),timing_files=files))
    write(root/'report.json',result);(root/'report.md').write_text(markdown(result));return result
