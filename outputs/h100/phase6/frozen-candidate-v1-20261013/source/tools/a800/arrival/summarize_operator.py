"""Require all ranks/shapes, audit actual kernels, and compare both arrival bases."""
import csv,json,math,random,re,statistics,sys
from pathlib import Path

def write(path,rows):
 with path.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def q(a,p):
 a=sorted(a);x=(len(a)-1)*p;i=int(x);return a[i]+(a[min(i+1,len(a)-1)]-a[i])*(x-i)
def paired(values):
 rng=random.Random(20261002);logs=[math.log(v) for v in values]
 boot=[100*(1-math.exp(statistics.mean(rng.choices(logs,k=len(logs))))) for _ in range(10000)]
 return 100*(1-math.exp(statistics.mean(logs))),q(boot,.025),q(boot,.975)
def result_record_names(case):
 return {p.name for p in case.glob('window-*-rank*.json') if re.fullmatch(r'window-\d+-.+-rank\d+\.json',p.name)}

def trial_maxima(rows,cfg):
 # Validate before max: max(1., nan) is 1., silently hiding a failed rank.
 assert cfg['trial_count']>0 and cfg['iters']>0
 protocols={r.get('timing_protocol','legacy') for r in rows}
 assert len(protocols)==1, 'Mixed timing protocols'
 for r in rows:
  assert len(r['trials_us'])==cfg['trial_count']
  assert all(math.isfinite(t) and t>0 for t in r['trials_us'])
  for key in ('iters','trial_count','warmup_initial','warmup_per_trial'):
   assert r[key]==cfg[key], ('Mismatched timing configuration',key)
 return [max(r['trials_us'][i] for r in rows) for i in range(cfg['trial_count'])]

def main(out):
 cfg=json.loads((out/'experiment.json').read_text());policies=cfg['policies'];orders=cfg['orders']
 assert len(set(policies))==len(policies)
 assert all(len(o)==len(policies) and set(o)==set(policies) for o in orders)
 windows=[];profiles=[];summaries=[];comparisons=[];checks=0;fp32_failures=0
 for m,n,k in cfg['shapes']:
  case=out/f'tp4-m{m}-n{n}-k{k}';ident=dict(M=m,N=n,K_global=k,TP=4,ring_reduction=True);block_times={}
  actual_order=[(b,p,False) for b,o in enumerate(orders) for p in o]+[(0,p,True) for p in policies]
  expected={f'window-{w:02d}-{p}-rank{r}.json' for w,(b,p,d) in enumerate(actual_order) for r in range(4)}
  actual=result_record_names(case)
  assert actual==expected,(case,sorted(expected-actual),sorted(actual-expected))
  signatures={}
  for window,(block,policy,diagnostic) in enumerate(actual_order):
   rows=[json.loads((case/f'window-{window:02d}-{policy}-rank{r}.json').read_text()) for r in range(4)]
   for rank,row in enumerate(rows):
    assert row['completed'] and row['rank']==rank and row['policy']==policy and row['block']==block and row['window']==window
    assert row['world_size']==4 and (row['M'],row['N'],row['K_global'])==(m,n,k)
    assert row['diagnostic_only']==diagnostic and not row['sampling']
    assert [c['seed'] for c in row['checks']]==cfg['seeds'] and all(c['passed'] for c in row['checks'])
    signature=(row['gpu_uuid'],row.get('timing_protocol','legacy'))
    assert signature==signatures.setdefault(rank,signature)
    checks+=len(row['checks'])
    fp32_failures+=sum(not c['fp32_passed'] for c in row['checks'])
    if diagnostic:
     assert not row['trials_us']
     es=json.loads((case/f'window-{window:02d}-{policy}-rank{rank}-profile.json').read_text())['traceEvents']
     kernels=[e for e in es if e.get('cat')=='kernel' and 'flux_bf16' in e['name']]
     assert len(kernels)==10
     log=(case/f'mapping-{policy}.log').read_text();grid=int(re.search(rf'PASS rank={rank} .*grid=(\d+)',log).group(1))
     for e in kernels:
      assert '64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic' in e['name']
      assert e['args']['grid']==[grid,1,1] and e['args']['block']==[128,1,1]
      assert 0<e['args']['registers per thread']<=255
     profiles.append(dict(ident,policy=policy,rank=rank,kernel_us=statistics.mean(e['dur'] for e in kernels),registers=kernels[0]['args']['registers per thread'],grid=grid))
   assert len({r['gpu_uuid'] for r in rows})==4
   if diagnostic:continue
   ts=trial_maxima(rows,cfg)
   median=statistics.median(ts)
   block_times[(block,policy)]=median
   windows.append(dict(ident,window=window,block=block,policy=policy,median_us=median,cv_percent=100*statistics.stdev(ts)/statistics.mean(ts) if len(ts)>1 else 0.,max_abs_vs_fp32=max(c['max_abs_vs_fp32'] for r in rows for c in r['checks'])))
  for policy in policies:
   with (case/f'mapping-{policy}.csv').open() as f:maps=list(csv.DictReader(f))
   tm,tn=m//128,(n+127)//128
   for rank in range(4):
    rr=[r for r in maps if int(r['rank'])==rank]
    assert len(rr)==tm*tn and {(int(r['m']),int(r['n'])) for r in rr}=={(i,j) for i in range(tm) for j in range(tn)}
   if not orders:continue
   times=[block_times[(b,policy)] for b in range(len(orders))]
   summaries.append(dict(ident,policy=policy,median_us=statistics.median(times),blocks=len(times)))
   for base in ['original','remote_first','interleaved']:
    ratios=[block_times[(b,policy)]/block_times[(b,base)] for b in range(len(orders))]
    mid,lo,hi=paired(ratios)
    comparisons.append(dict(ident,policy=policy,baseline=base,latency_reduction_percent=mid,ci95_low=lo,ci95_high=hi,faster_blocks=sum(v<1 for v in ratios),blocks=len(ratios)))
 if windows:write(out/'windows.csv',windows);write(out/'summary.csv',summaries);write(out/'comparisons.csv',comparisons)
 write(out/'profiles.csv',profiles)
 report=dict(completed=True,correctness_checks=checks,fp32_diagnostic_failures=fp32_failures,fp32_policy='diagnostic_only_user_authorized',summary=summaries,comparisons=comparisons,scope='Uninstrumented complete GemmRS, ring_reduction=True; separate profile. Intervals are within-job paired blocks.')
 (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main(Path(sys.argv[1]))
