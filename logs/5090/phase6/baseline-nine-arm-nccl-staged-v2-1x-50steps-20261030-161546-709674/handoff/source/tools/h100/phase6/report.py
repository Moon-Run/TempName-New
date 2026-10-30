"""Strict CPU verification and paired full-step H100 comparison statistics."""
import argparse
import csv
import gzip
import hashlib
import math
from pathlib import Path
import random
import statistics

from common import CONTROLS, PROTOCOL, orders, policy, read, require, sha, verify_files, write


def verify_inputs(root):
    info=read(root/'submission.json')
    require(info['protocol']==PROTOCOL,'Unknown run protocol')
    for base,items in ((root,info['files']),('/',info['source_manifests']),('/',info['build_files'])):
        verify_files(base,items)
    return info


def window(root,directory,expected_policy,mode,block=-1,index=-1):
    info=read(root/'submission.json');c=info['config'];world=c['tp']
    records=[read(directory/f'rank{r}.json') for r in range(world)]
    expected_model={k:c[k] for k in ('layers','hidden','ffn','heads','sequence','tp','micro_batch','global_batch','seed','token_seed','vocab')}
    expected_model['name']='h100-phase6'
    for rank,r in enumerate(records):
        require(r['completed'] and r['passed'] and r['skips']==0 and r['fallback_calls']==0,'Failed/incomplete/skipped window')
        require(r['run_id']==info['run_id'] and r['policy']==expected_policy and r['mode']==mode,'Window identity mismatch')
        require(r['rank']==r['local_rank']==r['tp_rank']==rank and r['dp_rank']==0,'Not a local TP group')
        require(r['tp_group_ranks']==list(range(world)) and r['dp_group_ranks']==[rank] and r['world_size']==world,'Topology mismatch')
        require(r['hardware']['host']==info['calibration_hardware']['host'] and r['hardware']['devices']==info['calibration_hardware']['devices'],'Recalibrate on this exact H100 allocation/device ordering')
        require(r['model']==expected_model,'Model configuration mismatch')
        require(r['block']==block and r['window']==index,'Pairing block mismatch')
        require(all(math.isfinite(x) for x in r['losses']+r['grad_norms']+[r['initial_loss']]),'Non-finite trajectory')
        if mode=='timing':
            require(r['timed_steps']==c['timed_steps'] and r['warmup_steps']==c['warmup_steps'],'Truncated measurement')
            require(len(r['losses'])==len(r['grad_norms'])==c['timed_steps'],'Incomplete trajectory')
            require(math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0,'Invalid timing')
        if expected_policy in ('native','native_taco'):
            require(r['flux_library_loaded'] is False,'Native baseline loaded Flux')
        else:
            for lib in ('libflux_cuda.so','libflux_cuda_ths_op.so'):
                path=(Path(info['variants'][expected_policy]['root'])/lib).resolve()
                require(r[lib]['path']==str(path) and r[lib]['sha256']==sha(path),'Wrong runtime library')
        if expected_policy=='native_taco':require(r['native_linear_forward_preserved'],'Native quantization replaced linear forward')
        if mode=='smoke' and expected_policy!='native':
            count=c['layers'] if expected_policy=='native_taco' else 2*c['layers']
            require(len(r['forward_checks'])==count,'Missing forward correctness checks')
            for check in r['forward_checks'].values():
                require(check.get('passed',True),'Codec error budget failed')
        if expected_policy not in ('native','original'):
            checks=r['selection_checks'];require(len(checks)==c['layers'],'Missing MLP wire checks')
            m=c['sequence']*c['micro_batch'];n=c['hidden'];bf16=(world-1)*(m//world)*n*2
            digest=None;wire=(world-1)*(m//world)*(n//128)*136
            if expected_policy.endswith('_selective'):
                base=info['variants'][expected_policy]['base']
                mask=read(root/'scripts/selection-plan.json')['policies'][base][0]['mask']
                digest=hashlib.sha256(bytes(v for row in mask for v in row)).hexdigest()
                wire=bf16+sum(mask[rank])*128*(136-256)
            for check in checks.values():
                require(check['mask_sha256']==digest and check['wire_bytes']==wire and check['bf16_wire_bytes']==bf16,'Wrong mask or logical communication volume')
    return records


def fingerprints(records):
    return [(r['initial_parameters_sha256'],r['initial_rng_sha256'],r['tokens_sha256'],
             r['parameter_count_local'],r['padded_vocab_size'],r['gpu_uuid'],
             r['torch'],r['cuda'],str(r['nccl']),r['python']) for r in records]


def preflight(root):
    info=verify_inputs(root);reference=None;profiles={}
    for p in info['policies']:
        rows=window(root,root/'results'/('smoke-'+p),p,'smoke')
        key=fingerprints(rows)
        if reference is None:reference=key
        require(key==reference,'Different initialization/input/hardware between policies')
        rows=window(root,root/'results'/('profile-'+p),p,'profile')
        require(fingerprints(rows)==reference,'Profile initialization differs')
        for rank in range(info['config']['tp']):
            path=root/'results'/('profile-'+p)/f'rank{rank}-profile.json.gz'
            with gzip.open(path,'rt') as f:
                import json
                events=json.load(f)['traceEvents']
            names=[e['name'] for e in events if e.get('cat')=='kernel']
            if p not in ('native','native_taco'):
                require(any('flux_bf16' in name and 'sm90' in name.lower() for name in names),'Hopper attention/reference kernel absent')
                if p!='original':
                    require(any('flux_bf16' in name and 'sm80' in name.lower() for name in names),'Phase6 V2 MLP kernel absent')
                    require(any('taco_decode' in name for name in names),'TACO decode kernel absent')
                    if p=='taco_fused' and info['variants'][p].get('implementation')=='a800-fused-v2-cta-v1':
                        require(any('taco_decode_ring_kernel(' in name for name in names),
                                'A800-aligned fused baseline must execute the non-templated CTA decoder')
            profiles[f'{p}/rank{rank}']=sorted(set(names))
    for name,v in info['variants'].items():
        if v['role']=='candidate':
            path=root/'results'/f'mapping-{name}.csv'
            require(path.is_file(),'Missing executed candidate mapping check')
            with path.open() as f:rows=[{k:int(v) for k,v in row.items()} for row in csv.DictReader(f)]
            entry=read(root/'scripts/selection-plan.json')['policies'][v['base']][0]
            c=info['config'];tiles=entry['tiles'];nt=c['hidden']//128
            require(len(rows)==c['tp']*tiles,'Incomplete GPU mapping output')
            for rank in range(c['tp']):
                rr=sorted((r for r in rows if r['rank']==rank),key=lambda r:r['index'])
                require([r['index'] for r in rr]==list(range(tiles)),'Missing/duplicate GPU mapping index')
                require([r['m']*nt+r['n'] for r in rr]==entry['coords'][rank],'Executed swizzle differs from the frozen plan')
    files={str(p.relative_to(root)):sha(p) for p in (root/'results').rglob('*') if p.is_file()}
    result=dict(completed=True,protocol=PROTOCOL,submission_sha256=sha(root/'submission.json'),
                fingerprints=reference,kernel_names=profiles,files=files)
    write(root/'preflight.json',result)
    return result


def paired(values,seed=20261009,resamples=10000):
    require(len(values)>=2 and all(math.isfinite(x) for x in values),'Insufficient/invalid paired blocks')
    gain=lambda xs:100*(1-math.exp(statistics.mean(xs)))
    rng=random.Random(seed)
    boot=sorted(gain([rng.choice(values) for _ in values]) for _ in range(resamples))
    return dict(percent=gain(values),ci95=[boot[int(.025*resamples)],boot[int(.975*resamples)]],blocks=len(values))


def report(root):
    info=verify_inputs(root);c=info['config'];gate=read(root/'preflight.json')
    require(gate['completed'] and gate['submission_sha256']==sha(root/'submission.json'),'Unverified run')
    verify_files(root,gate['files'])
    result=dict(protocol=PROTOCOL,completed=False,gpu_validated=True,rounds=[],target_percent=c['target_percent'],
                scope='H100 full optimizer steps only; SM90-compiled V2 quantized MLP vs native Hopper reference; no convergence claim.')
    result['fused_baseline_implementation']=info['variants']['taco_fused'].get('implementation','phase6-warp-v1')
    lines=['# H100 单节点 Phase6 完整 step 对比','',
           '原生 Flux 参考使用 Hopper V3；融合量化 baseline 与组合候选使用编译为 SM90 的 V2 MLP 路径。',
           f'融合基线实现：`{result["fused_baseline_implementation"]}`；不同基线版本的收益分开报告。',
           '正值为配对耗时下降；95% 区间按配对块 bootstrap。只报告本次同节点、同模型数据。','',
           '| 轮次 | 候选 | 对照 | 下降% | 95%区间 |','| --- | --- | --- | ---: | --- |']
    targets={policy(b):True for b in c['bases']}
    for rep,blocks in enumerate(orders(c),1):
        values={p:[] for p in info['policies']};index=0
        for block,order in enumerate(blocks):
            for p in order:
                folder=root/'results'/f'round{rep}'/f'window-{index:03d}-{p}'
                rows=window(root,folder,p,'timing',block,index)
                require(fingerprints(rows)==[tuple(x) for x in gate['fingerprints']],'Timing initialization/input/hardware changed')
                values[p].append(max(r['elapsed_seconds'] for r in rows)*1000/c['timed_steps'])
                index+=1
        comparisons={}
        for p in targets:
            comparisons[p]={}
            for base in CONTROLS:
                stats=paired([math.log(a/b) for a,b in zip(values[p],values[base])])
                comparisons[p][base]=stats
                if base in ('native_taco','taco_fused'):
                    targets[p] &= stats['percent']>=c['target_percent'] and stats['ci95'][0]>0
                low,high=stats['ci95']
                lines.append(f'| {rep} | {p} | {base} | {stats["percent"]:.3f} | [{low:.3f}, {high:.3f}] |')
        result['rounds'].append(dict(round=rep,median_ms={p:statistics.median(v) for p,v in values.items()},comparisons=comparisons))
    result.update(completed=True,accepted=targets)
    lines += ['',f'两轮均通过两个必须 baseline 的 {c["target_percent"]}% 门槛：`{targets}`。',
              '完整数值、每轮绝对耗时和区间见 report.json。']
    write(root/'report.json',result);(root/'report.md').write_text('\n'.join(lines)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('root',type=Path);p.add_argument('--preflight',action='store_true')
    a=p.parse_args();print(preflight(a.root.resolve()) if a.preflight else report(a.root.resolve()))
