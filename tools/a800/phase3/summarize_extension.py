"""Strict all-rank paired analysis. Keep regressions and every predeclared sample."""
import csv,json,math,random,re,statistics,sys
from pathlib import Path
from itertools import combinations

def quantile(values,p):
 values=sorted(values);x=(len(values)-1)*p;i=int(x)
 return values[i]+(values[min(i+1,len(values)-1)]-values[i])*(x-i)

def write(out,name,rows):
 with (out/name).open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def check_mapping_rows(rs,m,n,tp,rank,policy):
 tm,tn=m//128,(n+127)//128;tiles=tm*tn
 assert len(rs)==tiles and len({r['index'] for r in rs})==tiles
 assert {(r['m'],r['n']) for r in rs}=={(i,j) for i in range(tm) for j in range(tn)}
 assert all(r['destination']==r['m']//(tm//tp) for r in rs)
 if policy=='interleaved':
  assert {r['index'] for r in rs}==set(range(tiles)), 'Padded cohorts are not supported by this variant'
  assert all(r['destination']==(r['index']%tp+rank)%tp and
             r['m']==r['destination']*(tm//tp)+(r['index']//tp)//tn and
             r['n']==(r['index']//tp)%tn for r in rs)
 else:
  assert policy in ('original','remote_first')
  assert all(r['n']==r['base_n'] and r['m']==(r['base_m']+tm//tp*(rank+(policy=='remote_first')))%tm for r in rs)
 owners=[r['destination'] for r in sorted(rs,key=lambda r:r['index'])]
 assert all(owners.count(dest)==tiles//tp for dest in range(tp))
 return owners

def analyze(out):
 config=json.loads((out/'experiment.json').read_text());tp=config['world_size']
 policies=config['policies'];expected_windows=sum(len(o) for o in config['orders'])
 windows=[];paired=[];summary=[];mapping=[];stages=[];policy_pairs=[];checks=0
 for m,n,k in config['shapes']:
  case=out/f'tp{tp}-m{m}-n{n}-k{k}'
  ident=dict(TP=tp,M=m,N=n,K_global=k,K_local=k//tp)
  blocks={i:{} for i in range(len(config['orders']))}
  rank0=sorted(case.glob('window-*-rank0.json'))
  assert len(rank0)==expected_windows+len(policies),(case,len(rank0))
  all_records={}
  for path in rank0:
   records=[json.loads(path.with_name(path.name.replace('rank0.json',f'rank{r}.json')).read_text()) for r in range(tp)]
   d=records[0];window=d['window'];policy=d['policy'];block=d['block']
   assert window not in all_records;all_records[window]=records
   for rank,r in enumerate(records):
    assert r['completed'] and not r['sampling'] and r['rank']==rank and r['world_size']==tp
    for key in ['window','block','policy','M','N','K_global','K_local','diagnostic_only','job_id']:
     assert r[key]==d[key],(path,rank,key)
    assert (r['M'],r['N'],r['K_global'],r['K_local'])==(m,n,k,k//tp)
    assert [c['seed'] for c in r['checks']]==config['seeds'] and all(c['passed'] for c in r['checks'])
    assert all(r[key]==config[key] for key in ['warmup_initial','warmup_per_trial','iters','trial_count'])
    checks+=len(r['checks'])
   assert len({r['gpu_uuid'] for r in records})==tp
   if d['diagnostic_only']:
    assert window>=expected_windows and all(not r['trials_us'] for r in records)
    continue
   assert 0<=block<len(config['orders']) and policy in policies
   assert all(len(r['trials_us'])==config['trial_count'] for r in records)
   assert all(math.isfinite(t) and t>0 for r in records for t in r['trials_us'])
   values=[max(r['trials_us'][i] for r in records) for i in range(config['trial_count'])]
   row=dict(ident,window=window,block=block,policy=policy,median_us=statistics.median(values),
       min_us=min(values),max_us=max(values),cv_percent=100*statistics.stdev(values)/statistics.mean(values),
       max_abs_vs_fp32=max(c['max_abs_vs_fp32'] for r in records for c in r['checks']),
       max_relative_l2_vs_fp32=max(c['relative_l2_vs_fp32'] for r in records for c in r['checks']))
   assert policy not in blocks[block];blocks[block][policy]=row;windows.append(row)
  assert set(all_records)==set(range(expected_windows+len(policies)))
  assert len({tuple(r['gpu_uuid'] for r in rs) for rs in all_records.values()})==1
  for block,order in enumerate(config['orders']):
   assert set(blocks[block])==set(policies)
   assert [p for p,_ in sorted(blocks[block].items(),key=lambda item:item[1]['window'])]==order
  for policy in policies:
   latencies=[b[policy]['median_us'] for b in blocks.values()]
   ratios=[b['original']['median_us']/b[policy]['median_us'] for b in blocks.values()]
   for block,ratio in enumerate(ratios):
    paired.append(dict(ident,block=block,policy=policy,original_us=blocks[block]['original']['median_us'],
       candidate_us=blocks[block][policy]['median_us'],speedup=ratio,latency_reduction_percent=100*(1-1/ratio)))
   logs=list(map(math.log,ratios));rng=random.Random(20261001)
   boot=[math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000)]
   speedup=math.exp(statistics.mean(logs))
   summary.append(dict(ident,policy=policy,median_window_us=statistics.median(latencies),
       paired_geomean_speedup=speedup,paired_latency_reduction_percent=100*(1-1/speedup),
       faster_blocks=sum(r>1 for r in ratios),total_blocks=len(ratios),
       bootstrap_low=quantile(boot,.025),bootstrap_high=quantile(boot,.975),
       max_window_cv_percent=max(b[policy]['cv_percent'] for b in blocks.values()),correctness='PASS'))
   with (case/f'mapping-{policy}.csv').open() as f:rows=[{key:int(v) for key,v in r.items()} for r in csv.DictReader(f)]
   tm,tn=m//128,(n+127)//128;tiles=tm*tn
   assert len(rows)==tp*tiles
   for rank in range(tp):
    rs=[r for r in rows if r['rank']==rank]
    owners=check_mapping_rows(rs,m,n,tp,rank,policy)
    mapping.append(dict(ident,policy=policy,rank=rank,tiles=tiles,local_tiles=owners.count(rank),
        remote_tiles=tiles-owners.count(rank),initial_remote_tiles=next((i for i,o in enumerate(owners) if o==rank),tiles),
        destination_transitions=sum(a!=b for a,b in zip(owners,owners[1:]))))
  for reference,candidate in combinations(policies,2):
   ratios=[b[reference]['median_us']/b[candidate]['median_us'] for b in blocks.values()]
   logs=list(map(math.log,ratios));rng=random.Random(20261001)
   boot=[math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000)]
   speedup=math.exp(statistics.mean(logs))
   policy_pairs.append(dict(ident,reference=reference,candidate=candidate,
       paired_geomean_speedup=speedup,paired_latency_reduction_percent=100*(1-1/speedup),
       faster_blocks=sum(r>1 for r in ratios),total_blocks=len(ratios),
       bootstrap_low=quantile(boot,.025),bootstrap_high=quantile(boot,.975)))
  case_stages=[]
  for path in sorted(case.glob('*-profile.json')):
   match=re.fullmatch(r'window-(\d+)-(.+)-rank(\d+)-profile.json',path.name);assert match
   window,policy,rank=match.groups();assert all_records[int(window)][int(rank)]['diagnostic_only']
   gpu=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat') in ('kernel','gpu_memset')]
   groups={name:[] for name in ['fused_gemm_store','ipc_barrier','local_reduce','memset']}
   for e in gpu:
    name=e['name']
    group=('fused_gemm_store' if 'flux_bf16' in name else 'ipc_barrier' if 'CudaIpcBarrierAllKernel' in name
           else 'local_reduce' if 'at::native::reduce_kernel' in name else 'memset' if e['cat']=='gpu_memset' else None)
    if group:groups[group].append(e)
   assert all(len(es)==10 for es in groups.values()),(path,{g:len(es) for g,es in groups.items()})
   kernel=groups['fused_gemm_store'][0]
   assert 'rcr_gemmv2_false_false_intranode_' in kernel['name']
   h=re.search(r'64x64x32_16x8x16_streamksk_nil_\d+x\d+x\d+_gemmstreamk_\d+_rasterheuristic',kernel['name']);assert h
   assert h.group()=='64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic',(path,h.group())
   row=dict(ident,policy=policy,rank=int(rank),**{g+'_us':statistics.mean(e['dur'] for e in es) for g,es in groups.items()},
     registers_per_thread=kernel['args']['registers per thread'],shared_memory=kernel['args']['shared memory'],
     grid=str(kernel['args']['grid']),block=str(kernel['args']['block']),hparams=h.group())
   log=(case/f'mapping-{policy}.log').read_text()
   expected_grid=int(re.search(rf'PASS rank={rank} .*grid=(\d+)',log).group(1))
   assert kernel['args']['grid']==[expected_grid,1,1],(path,expected_grid,kernel['args']['grid'])
   case_stages.append(row)
  assert len(case_stages)==len(policies)*tp
  assert {(r['policy'],r['rank']) for r in case_stages}=={(p,r) for p in policies for r in range(tp)}
  assert len({(r['registers_per_thread'],r['shared_memory'],r['grid'],r['block'],r['hparams']) for r in case_stages})==1
  stages.extend(case_stages)
 for name,rows in [('windows.csv',windows),('paired_blocks.csv',paired),('summary.csv',summary),('mapping_summary.csv',mapping),('diagnostic_stages.csv',stages),('policy_pairs.csv',policy_pairs)]:write(out,name,rows)
 report=dict(summary=summary,correctness_checks_including_diagnostics=checks,
   limitations=['Six paired blocks per shape; bootstrap intervals describe this job only.',
    'TP changes local K; compare candidates within each TP, not absolute TP2 versus TP4 latency.',
    'Rank+1 rotation follows the original raster: remote-first is not a universal whole-grid ordering guarantee.',
    'M partitions remain 128-row aligned; N tail covered, unaligned M partitions not yet supported by this experiment.',
    'Profiling follows all official timing; cache counters and tile arrival metrics remain unmeasured.']+
    (['Interleaved cycles all TP destinations, preserving a 1:3 local/remote ratio at TP4. It changes within-partition locality to row-major; padded cohorts are rejected.'] if 'interleaved' in policies else []))
 (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
 print(json.dumps(report,indent=2))
 return report

if __name__=='__main__':analyze(Path(sys.argv[1]))
