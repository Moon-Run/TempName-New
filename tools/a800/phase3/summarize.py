"""Analyze six balanced policy orders; windows, not individual iterations, are comparison units."""
import csv,json,math,random,re,statistics,sys
from collections import defaultdict
from pathlib import Path
out=Path(sys.argv[1]);policies=['original','remote_first','interleaved']
def med(v):return statistics.median(v)
def q(v,p):
 v=sorted(v);x=(len(v)-1)*p;i=int(x);return v[i]+(v[min(i+1,len(v)-1)]-v[i])*(x-i)
def write(name,rows):
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
windows=[];checks=0;blocks=defaultdict(dict)
files=sorted(out.glob('window-*-rank0.json'));assert len(files)==18,len(files)
for path in files:
 d=json.loads(path.read_text());peer=json.loads(Path(str(path).replace('rank0','rank1')).read_text())
 assert d['completed'] and peer['completed'] and d['policy']==peer['policy']
 assert d['M']==d['N']==4096 and d['K_global']==8192 and not d['sampling']
 for r in [d,peer]:assert len(r['checks'])==3 and all(c['passed'] for c in r['checks']);checks+=3
 values=[max(a,b) for a,b in zip(d['trials_us'],peer['trials_us'])];assert len(values)==12
 row=dict(window=d['window'],block=d['block'],policy=d['policy'],median_us=med(values),
          min_us=min(values),max_us=max(values),cv_percent=100*statistics.stdev(values)/statistics.mean(values),
          max_abs_vs_fp32=max(c['max_abs_vs_fp32'] for r in [d,peer] for c in r['checks']),
          max_relative_l2_vs_fp32=max(c['relative_l2_vs_fp32'] for r in [d,peer] for c in r['checks']))
 windows.append(row);assert d['policy'] not in blocks[d['block']];blocks[d['block']][d['policy']]=row
assert len(blocks)==6 and all(set(b)==set(policies) for b in blocks.values())
orders=[[r['policy'] for r in sorted(windows,key=lambda r:r['window']) if r['block']==b] for b in sorted(blocks)]
assert len(set(map(tuple,orders)))==6
paired=[];summary=[]
for policy in policies:
 latencies=[blocks[b][policy]['median_us'] for b in sorted(blocks)]
 ratios=[blocks[b]['original']['median_us']/blocks[b][policy]['median_us'] for b in sorted(blocks)]
 for b,ratio in zip(sorted(blocks),ratios):
  paired.append(dict(block=b,policy=policy,original_us=blocks[b]['original']['median_us'],
                     candidate_us=blocks[b][policy]['median_us'],speedup=ratio,latency_reduction_percent=100*(1-1/ratio)))
 logs=[math.log(r) for r in ratios];rng=random.Random(20261001)
 boot=[math.exp(statistics.mean(rng.choices(logs,k=6))) for _ in range(10000)]
 g=math.exp(statistics.mean(logs))
 summary.append(dict(policy=policy,median_window_us=med(latencies),min_window_us=min(latencies),max_window_us=max(latencies),
    paired_geomean_speedup=g,paired_latency_reduction_percent=100*(1-1/g),
    paired_speedup_min=min(ratios),paired_speedup_max=max(ratios),faster_blocks=sum(r>1 for r in ratios),
    bootstrap_low=q(boot,.025),bootstrap_high=q(boot,.975),correctness='PASS'))
write('windows.csv',windows);write('paired_blocks.csv',paired);write('summary.csv',summary)
# Exact compiled mapping on GPU, including endpoints, not a mirror-only CPU test.
mappings={}
for policy in policies:
 rows=[list(map(int,line.split(','))) for line in (out/f'mapping-{policy}.csv').read_text().splitlines()]
 assert len(rows)==2048
 stats=[]
 for rank in range(2):
  rs=[r for r in rows if r[0]==rank];assert len(rs)==1024
  assert {(r[2],r[3]) for r in rs}=={(m,n) for m in range(32) for n in range(32)}
  assert all(r[4]==r[2]//16 for r in rs)
  if policy=='interleaved':assert all(r[4]==(r[1]%2+rank)%2 for r in rs)
  if policy=='original':assert all(r[4]==rank for r in rs[:512])
  if policy=='remote_first':assert all(r[4]!=rank for r in rs[:512])
  owners=[r[4] for r in rs]
  stats.append(dict(rank=rank,tiles=1024,local_tiles=sum(o==rank for o in owners),
    destination_transitions=sum(a!=b for a,b in zip(owners,owners[1:]))))
 mappings[policy]=stats
(out/'mapping_summary.json').write_text(json.dumps(mappings,indent=2)+'\n')
# Profiles are explicitly separate from formal timing; preserve resource metadata.
stages=[];hparams=set()
for path in sorted(out.glob('*-profile.json')):
 match=re.match(r'window-\d+-(.+)-rank(\d+)-profile.json',path.name);policy,rank=match.groups()
 gpu=[e for e in json.loads(path.read_text())['traceEvents'] if e.get('cat') in ('kernel','gpu_memset')]
 groups=defaultdict(list)
 for e in gpu:
  name=e['name']
  if 'flux_bf16' in name:group='fused_gemm_store'
  elif 'CudaIpcBarrierAllKernel' in name:group='ipc_barrier'
  elif 'at::native::reduce_kernel' in name:group='local_reduce'
  elif e['cat']=='gpu_memset':group='memset'
  else:continue  # profiler setup fill + NCCL barrier
  groups[group].append(e)
 for g in ['fused_gemm_store','ipc_barrier','local_reduce','memset']:assert len(groups[g])==10,(path,g,len(groups[g]))
 kernel=groups['fused_gemm_store'][0]
 h=re.search(r'64x64x32_16x8x16_streamksk_nil_\d+x\d+x\d+_gemmstreamk_\d+_rasterheuristic',kernel['name']);assert h
 hparams.add(h.group())
 stages.append(dict(policy=policy,rank=int(rank),**{g+'_us':statistics.mean(e['dur'] for e in es) for g,es in groups.items()},
                   registers_per_thread=kernel['args']['registers per thread'],shared_memory=kernel['args']['shared memory'],
                   grid=str(kernel['args']['grid']),block=str(kernel['args']['block']),hparams=h.group()))
assert len(stages)==6 and len(hparams)==1
assert next(iter(hparams))=='64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic'
write('diagnostic_stages.csv',stages)
report=dict(summary=summary,checks=checks,orders=orders,hparams=next(iter(hparams)),mappings=mappings,
 limitations=['One fixed shape, TP2, same allocated node and GPUs per job.',
 'Policy windows are separate processes, balanced across all six permutations; not simultaneous comparisons.',
 'Bootstrap resamples six window blocks and is descriptive; it does not establish cross-workload generality.',
 'Profile timings are diagnostic only and excluded from headline performance.',
 'Logical tile order does not guarantee physical CTA completion order.'])
(out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
