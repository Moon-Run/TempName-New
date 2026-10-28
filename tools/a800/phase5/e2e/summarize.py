"""All-rank model metrics and paired-window analysis for TP=4/8 scale cases."""
import csv,gzip,hashlib,json,math,random,re,statistics,sys
from collections import Counter
from pathlib import Path
from routing import module_routes, codec_placement, base_policy, selection_entry, logical_bytes

def write(path,rows):
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def q(xs,p):
    xs=sorted(xs);x=(len(xs)-1)*p;i=int(x)
    return xs[i]+(xs[min(i+1,len(xs)-1)]-xs[i])*(x-i)

def expected_decoder_counts(plan, selection, model, policy, rank, calls):
    """Predict exact launches from the frozen mask; never infer them from a trace."""
    result = dict(legacy=0, bf16=0, compact=0)
    if codec_placement(policy) not in ('fused','separate'):
        return result
    invocations = calls // 2
    m, n, world = model['sequence'] * model['micro_batch'], model['hidden'], model['tp']
    if (plan.get('selective_decoder') == 'compact-v1' and policy.endswith('_selective') and
            (m,n,world) == (8192,2048,4)):
        mask = selection_entry(selection,policy,m,n,model['ffn'])['mask']
        count = sum(any(mask[src][tile] for src in range(world))
                    for tile in range(rank*256,(rank+1)*256))
        if count <= 32:
            result.update(bf16=invocations,compact=invocations if count else 0)
            return result
    result['legacy'] = invocations
    return result

def analyze(out):
    selection=json.loads((out/'scripts/selection-plan.json').read_text())
    protocol=json.loads((out/'protocol.json').read_text());plan=protocol['plan'];case=protocol['case'];model=protocol['model']
    assert all(len(o)==len(plan['policies']) and set(o)==set(plan['policies']) for o in plan['orders'])
    tp=case['tp'];policies=plan['policies'];calls=model['layers']*2*(model['global_batch']//model['micro_batch'])
    tokens=model['sequence']*model['global_batch']
    m=model['sequence']*model['micro_batch'];n=model['hidden']
    shapes={(m,n,k) for k in [model['hidden'],model['ffn']]}
    def load(name,policy,mode):
        path=out/name
        assert (path/'exit-code.txt').read_text().strip()=='0'
        rows=[json.loads((path/f'rank{r}.json').read_text()) for r in range(tp)]
        for rank,r in enumerate(rows):
            assert r['completed'] and r['passed'] and r['rank']==rank and r['world_size']==tp
            assert r['case']==case['name'] and r['model']==model and r['mode']==mode and r['policy']==policy
            if mode!='profile':
                assert math.isfinite(r['elapsed_seconds']) and r['elapsed_seconds']>0
            assert r['fallback_calls']==0 and len(r['target_modules'])==2*model['layers']
            assert r['module_policies']==module_routes(r['observed_shapes'], policy, model['layers'],model)
            assert r['codec_placements']==({} if policy=='native' else {name: (codec_placement(policy) if name.endswith('.mlp.linear_fc2') else 'native' if policy=='native_taco' else 'bf16') for name in r['target_modules']})
            checks=r['selection_checks']
            if codec_placement(policy) in ('fused','separate','tensor'):
                assert len(checks)==model['layers']
                mask=selection_entry(selection,policy,m,n,model['ffn'])['mask'] if policy.endswith('_selective') else None
                digest=hashlib.sha256(bytes(v for row in mask for v in row)).hexdigest() if mask is not None else None
                expected_wire,bf16_wire=logical_bytes(mask,rank,m,n,tp)
                assert all(name.endswith('.mlp.linear_fc2') and c['mask_sha256']==digest and c['wire_bytes']==expected_wire and c['bf16_wire_bytes']==bf16_wire for name,c in checks.items())
            else:assert not checks
            assert r['tokens_per_optimizer_step']==tokens and not r['sampling']
            assert all(math.isfinite(x) for x in r['losses']+r['grad_norms']+[r['initial_loss']]) and r['skips']==0
            assert {(s['M'],s['N'],s['K_global']) for s in r['observed_shapes'].values()}==shapes
        assert len({r.get('timing_protocol','legacy') for r in rows})==1
        assert len({r['gpu_uuid'] for r in rows})==tp and len({r['tokens_sha256'] for r in rows})==1
        return rows
    if protocol['stage']=='preflight':
        profiles=[];smoke=[]
        for policy in policies:
            rows=load('smoke-'+policy,policy,'smoke')
            if policy != 'native':
                assert all(len(r['forward_checks'])==(model['layers'] if policy=='native_taco' else 2*model['layers']) for r in rows)
            elapsed=max(r['elapsed_seconds'] for r in rows)
            smoke.append(dict(policy=policy,smoke_ms_per_step=elapsed*1000/plan['smoke_steps'],
                initial_loss=rows[0]['initial_loss'],final_loss=rows[0]['losses'][-1],
                max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows)))
            rs=load('profile-'+policy,policy,'profile')
            for rank,r in enumerate(rs):
                with gzip.open(out/f'profile-{policy}/rank{rank}-profile.json.gz','rt') as f:
                    events=json.load(f)['traceEvents']
                kernels=[e for e in events if e.get('cat')=='kernel' and 'flux_bf16' in e['name']]
                assert len(kernels)==(0 if policy in ('native','native_taco') else calls),(policy,rank,len(kernels),calls)
                hparams=set()
                for e in kernels:
                    h=re.search(r'64x64x32_16x8x16_streamksk_nil_\d+x\d+x\d+_gemmstreamk_\d+_rasterheuristic',e['name']);assert h
                    hparams.add(h.group())
                    assert 0 < e['args']['registers per thread'] <= 255 and e['args']['block']==[128,1,1]
                if kernels:
                    assert hparams=={'64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic'}
                all_kernels=[e for e in events if e.get('cat')=='kernel']
                if plan.get('double_buffered') and policy.endswith('_selective'):
                    assert sum('CudaIpcBarrierAllKernel' in e['name'] for e in all_kernels)==calls, (policy,rank,'unexpected barrier count')
                if policy=='native_taco':
                    for label in ('encode','exchange','decode_reduce'):
                        assert sum(e.get('cat')=='user_annotation' and e.get('ph')=='X' and e.get('name')=='megatron_taco.'+label for e in events)==calls//2

                expected_decoders = expected_decoder_counts(plan,selection,model,policy,rank,calls)
                actual_decoders = dict(
                    legacy=sum(any(name in e['name'] for name in ('taco_decode_ring_kernel','taco_decode_tile4_kernel')) for e in all_kernels),
                    bf16=sum('taco_reduce_bf16_compact4_kernel' in e['name'] for e in all_kernels),
                    compact=sum('taco_decode_compact4_kernel' in e['name'] for e in all_kernels))
                assert actual_decoders == expected_decoders, (policy,rank,actual_decoders,expected_decoders)
                assert not any(any(name in e['name'] for name in ('taco_decode_pair4_kernel','taco_decode_flat4_kernel')) for e in all_kernels), 'Exploratory decoder used in formal run'
                assert sum('taco_encode_scatter_kernel' in e['name'] for e in all_kernels)==(calls//2 if policy=='taco_separate' else 0)
                profiles.append(dict(policy=policy,rank=rank,calls=len(kernels),hparams=';'.join(hparams),
                    kernel_sum_us=sum(e['dur'] for e in kernels),peak_allocated_gib=r['peak_allocated_gib']))
        write(out/'smoke.csv',smoke);write(out/'profile_summary.csv',profiles)
        result=dict(completed=True,case=case,model=model,script_sha256=protocol['script_sha256'],
            flux_manifest_sha256=protocol['flux_manifest_sha256'],smoke=smoke,profiles=profiles,
            scope='Finite full-model short training, actual shape/mapping/kernel checks; no small-model error thresholds.')
        (out/'preflight.json').write_text(json.dumps(result,indent=2)+'\n')
    else:
        windows=[];blocks={};signatures={};window=0
        assert len(list(out.glob('window-*')))==len(plan['orders'])*len(policies)
        for block,order in enumerate(plan['orders']):
            blocks[block]={}
            for policy in order:
                rows=load(f'window-{window:02d}-{policy}',policy,'timing')
                for rank,r in enumerate(rows):
                    assert r['window']==window and r['block']==block and not r['profiling']
                    assert r['warmup_steps']==plan['warmup_steps'] and r['timed_steps']==plan['timed_steps']
                    assert len(r['losses'])==len(r['grad_norms'])==plan['timed_steps']
                    signature=tuple(r[key] for key in ['initial_parameters_sha256','initial_rng_sha256','tokens_sha256','gpu_uuid'])+(r.get('timing_protocol','legacy'),)
                    if rank not in signatures:signatures[rank]=signature
                    assert signature==signatures[rank]
                    assert all(v==plan['warmup_steps']*model['global_batch']//model['micro_batch'] for v in r['warmup_calls'].values())
                elapsed=max(r['elapsed_seconds'] for r in rows)
                assert elapsed>0 and math.isfinite(elapsed)
                r=rows[0]
                row=dict(case=case['name'],TP=tp,hidden=model['hidden'],sequence=model['sequence'],window=window,block=block,policy=policy,
                    ms_per_step=1000*elapsed/plan['timed_steps'],tokens_per_second=tokens*plan['timed_steps']/elapsed,
                    initial_loss=r['initial_loss'],first_timed_loss=r['losses'][0],last_timed_loss=r['losses'][-1],
                    final_grad_norm=r['grad_norms'][-1],peak_allocated_gib=max(r['peak_allocated_gib'] for r in rows),
                    peak_reserved_gib=max(r['peak_reserved_gib'] for r in rows),skips=0,finite=True)
                windows.append(row);blocks[block][policy]=row;window+=1
        summary=[];paired=[]
        for policy in policies:
            points=[b[policy] for b in blocks.values()];times=[r['ms_per_step'] for r in points]
            row=dict(case=case['name'],TP=tp,hidden=model['hidden'],sequence=model['sequence'],policy=policy,
                median_ms_per_step=statistics.median(times),median_tokens_per_second=statistics.median(r['tokens_per_second'] for r in points),
                min_ms=min(times),max_ms=max(times),cv_percent=100*statistics.stdev(times)/statistics.mean(times) if len(times)>1 else 0.,
                initial_loss=statistics.median(r['initial_loss'] for r in points),first_timed_loss=statistics.median(r['first_timed_loss'] for r in points),
                last_timed_loss=statistics.median(r['last_timed_loss'] for r in points),final_grad_norm=statistics.median(r['final_grad_norm'] for r in points),
                max_peak_allocated_gib=max(r['peak_allocated_gib'] for r in points),max_peak_reserved_gib=max(r['peak_reserved_gib'] for r in points))
            for baseline in policies:
                logs=[math.log(b[policy]['ms_per_step']/b[baseline]['ms_per_step']) for b in blocks.values()]
                ratio=math.exp(statistics.mean(logs));rng=random.Random(20261002)
                boot=[100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(10000)]
                row.update({f'latency_reduction_vs_{baseline}_percent':100*(1-ratio),f'ci95_low_vs_{baseline}':q(boot,.025),f'ci95_high_vs_{baseline}':q(boot,.975),f'faster_blocks_vs_{baseline}':sum(x<0 for x in logs)})
                for block,b in blocks.items():paired.append(dict(block=block,policy=policy,baseline=baseline,baseline_ms=b[baseline]['ms_per_step'],candidate_ms=b[policy]['ms_per_step'],latency_reduction_percent=100*(1-b[policy]['ms_per_step']/b[baseline]['ms_per_step'])))
            summary.append(row)
        write(out/'windows.csv',windows);write(out/'summary.csv',summary);write(out/'paired_blocks.csv',paired)
        result=dict(completed=True,case=case,model=model,summary=summary,windows=len(windows),tokens_per_optimizer_step=tokens,
            timing_steps=[plan['warmup_steps']+1,plan['warmup_steps']+plan['timed_steps']],
            scope=plan['scope'],interval_unit=f"{len(plan['orders'])} paired blocks within this job")
        (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':analyze(Path(sys.argv[1]))
