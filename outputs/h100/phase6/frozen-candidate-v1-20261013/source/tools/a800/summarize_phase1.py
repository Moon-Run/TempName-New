"""Summarize phase1 results without running GPU work."""
import csv
import json
from pathlib import Path
import statistics
import sys
from collections import defaultdict

out=Path(sys.argv[1])
data=json.loads((out/'baseline.json').read_text())
assert data.get('completed') and len(data['shapes'])==6

def write_csv(name,rows):
    with (out/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

summary=[]
for row in sorted(data['shapes'],key=lambda r:(r['K_global'],r['M'])):
    checks=[r for s in row['checks'] for r in s['ranks']]
    assert len(checks)==6 and all(c['passed'] for c in checks)
    assert len(row['trials'])==12 and all(len(t['flux'])==len(t['torch'])==2 for t in row['trials'])
    summary.append(dict(M=row['M'],N=row['N'],K_global=row['K_global'],K_local=row['K_local'],
        flux_us=row['flux']['median_ms']*1000,torch_us=row['torch']['median_ms']*1000,
        speedup=row['speedup'],flux_cv_percent=row['flux']['cv_percent'],
        torch_cv_percent=row['torch']['cv_percent'],flux_iqr_us=row['flux']['iqr_ms']*1000,
        torch_iqr_us=row['torch']['iqr_ms']*1000,
        max_abs_vs_fp32=max(c['max_abs_vs_fp32'] for c in checks),
        max_relative_l2_vs_fp32=max(c['relative_l2_vs_fp32'] for c in checks),
        correctness='PASS',stability='HIGH_VARIANCE' if max(row['flux']['cv_percent'],row['torch']['cv_percent'])>5 else 'WITHIN_5_PERCENT_CV'))
write_csv('summary.csv',summary)

stages=[];kernels=[]
for profile in data['profiles']:
    for rank in range(2):
        path=out/f'profile-m{profile["M"]}-n{profile["N"]}-k{profile["K_global"]}-rank{rank}.json'
        raw=json.loads(path.read_text())
        gpu=sorted([e for e in raw['traceEvents'] if e.get('cat') in ('kernel','gpu_memset','gpu_memcpy')],key=lambda e:e['ts'])
        # The initial fill+NCCL barrier synchronizes profiler setup, before the timed forward loop.
        setup_nccl=[e for e in gpu if 'nccl' in e['name'].lower()]
        assert len(setup_nccl)==1
        begin=setup_nccl[0]['ts']+setup_nccl[0]['dur']
        measured=[e for e in gpu if e['ts']>=begin]
        groups=defaultdict(list)
        for event in measured:
            name=event['name']
            if event['cat']=='gpu_memset':group='workspace_memset'
            elif 'flux_bf16' in name:group='fused_gemm_remote_store'
            elif 'CudaIpcBarrierAllKernel' in name:group='ipc_barrier'
            elif 'at::native::reduce_kernel' in name:group='final_local_reduce'
            else:group='other'
            groups[group].append(event)
        for group in ['fused_gemm_remote_store','ipc_barrier','final_local_reduce','workspace_memset']:
            assert len(groups[group])==profile['iters'],(path,group,len(groups[group]))
        assert not groups.get('other'),(path,'unclassified GPU activity')
        assert len({e['args']['stream'] for e in measured})==1, 'Cannot sum overlapping streams'
        totals={group:sum(e['dur'] for e in events)/profile['iters'] for group,events in groups.items()}
        span=(max(e['ts']+e['dur'] for e in measured)-min(e['ts'] for e in measured))/profile['iters']
        unprofiled=statistics.mean([profile['unprofiled_before_ms'][rank],profile['unprofiled_after_ms'][rank]])*1000
        profiled=profile['profiled_ms'][rank]*1000
        stages.append(dict(M=profile['M'],N=profile['N'],K_global=profile['K_global'],rank=rank,
            iters=profile['iters'],**{g+'_us':totals[g] for g in sorted(totals)},
            gpu_activity_span_us=span,gaps_within_span_us=span-sum(totals.values()),
            matched_unprofiled_us=unprofiled,profiled_event_us=profiled,
            profiler_overhead_percent=100*(profiled/unprofiled-1),
            excluded_setup_nccl_us=setup_nccl[0]['dur']))
        for group,events in groups.items():
            kernels.append(dict(K_global=profile['K_global'],rank=rank,stage=group,count=len(events),
                mean_us=statistics.mean(e['dur'] for e in events),min_us=min(e['dur'] for e in events),
                max_us=max(e['dur'] for e in events),kernel_name=events[0]['name']))
write_csv('stage_summary.csv',stages);write_csv('kernel_summary.csv',kernels)
(out/'analysis.json').write_text(json.dumps(dict(baseline=summary,stages=stages),indent=2)+'\n')
print(json.dumps(dict(summary=summary,stages=stages),indent=2))
