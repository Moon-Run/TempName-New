"""Analyze four-source observations with explicit perturbation and censoring limits."""
import csv,json,math,re,statistics,sys
from collections import defaultdict
from pathlib import Path

def med(x):return statistics.median(x) if x else None
def q(x,p):
 if not x:return None
 s=sorted(x);v=(len(s)-1)*p;i=int(v);return s[i]+(s[min(i+1,len(s)-1)]-s[i])*(v-i)
def cv(x):return 100*statistics.stdev(x)/statistics.mean(x) if len(x)>1 else 0

def join_observation(records,world,fragments,poll_ns,origin,end):
 sources=defaultdict(dict)
 for r in records:
  assert r['fragment'] not in sources[r['source']]
  sources[r['source']][r['fragment']]=r['observed_ns']
 assert set(sources)==set(range(world))
 assert all(set(v)==set(range(fragments)) for v in sources.values())
 if any(t<0 for v in sources.values() for t in v.values()):return None
 times={src:max(v.values()) for src,v in sources.items()};order=sorted(times,key=times.get)
 last=times[order[-1]];margin=last-times[order[-2]]
 return dict(last_source=order[-1],last_source_resolved=margin>2*poll_ns,
             last_margin_us=margin/1000,join_skew_us=(last-min(times.values()))/1000,
             last_after_origin_us=(last-origin)/1000,forward_span_us=(end-origin)/1000,
             end_minus_last_us=(end-last)/1000,poll_cycle_us=poll_ns/1000,
             **{f'source{src}_after_origin_us':(t-origin)/1000 for src,t in times.items()})

def write(out,name,rows):
 if not rows:return
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def analyze(out):
 cfg=json.loads((out/'experiment.json').read_text());world=cfg['world_size'];smoke=cfg['smoke']
 orders=cfg['orders'][:1] if smoke else cfg['orders'];repeats=2 if smoke else cfg['repeats']
 data={};builds=[];perf=[];receivers=[];producers=[];validity=[];checks=0;identities={}
 for block,order in enumerate(orders):
  for policy in order:
   for kind in ['vanilla_before','instrumented','vanilla_after']:
    ranks=[json.loads((out/f'b{block}-{policy}-{kind}-rank{r}.json').read_text()) for r in range(world)]
    assert all(d['completed'] and d['rank']==r and d['world_size']==world and d['block']==block and d['policy']==policy and d['kind']==kind for r,d in enumerate(ranks))
    assert all(d.get('sample_warmup',0)==cfg.get('sample_warmup',0) for d in ranks)
    identities[(block,policy,kind)]=tuple(d['gpu_uuid'] for d in ranks)
    assert len(set(identities[(block,policy,kind)]))==world
    data[(block,policy,kind)]=ranks
    for si,(m,n,k) in enumerate(cfg['shapes']):
     ident=dict(block=block,policy=policy,M=m,N=n,K_global=k)
     rows=[d['shapes'][si] for d in ranks]
     assert all(r['correctness'] and (r['M'],r['N'],r['K_global'])==(m,n,k) for r in rows)
     times=[max(r['batch_us'][i] for r in rows) for i in range(2 if smoke else cfg['trials'])]
     assert all(math.isfinite(t) and t>0 for t in times)
     meta=[]
     for r in rows:
      ker=r['kernel'];h=re.search(r'64x64x32_16x8x16_streamksk_nil_\d+x\d+x\d+_gemmstreamk_\d+_rasterheuristic',ker['name'])
      assert h and h.group()=='64x64x32_16x8x16_streamksk_nil_128x128x32_gemmstreamk_3_rasterheuristic'
      meta.append((h.group(),str(ker['args']['grid']),str(ker['args']['block']),ker['args']['registers per thread']))
     assert len(set(meta))==1
     builds.append(dict(ident,kind=kind,median_us=med(times),cv_percent=cv(times),hparams=meta[0][0],grid=meta[0][1],threads=meta[0][2],registers=meta[0][3]))
     if kind!='instrumented':continue
     assert all(len(r['samples'])==repeats*len(cfg['modes']) for r in rows)
     for mode in cfg['modes']:
      by_rep=[];deltas=[];overheads=[];extras=[];off_times=[]
      for repeat in range(repeats):
       samples=[next(s for s in r['samples'] if s['repeat']==repeat and s['mode']==mode) for r in rows]
       controls=[next(s for s in r['samples'] if s['repeat']==repeat and s['mode']=='off') for r in rows]
       v=max(s['forward_us'] for s in samples);base=max(s['forward_us'] for s in controls)
       by_rep.append(v);off_times.append(base);deltas.append(v-base);overheads.append(100*(v/base-1));extras.append(max(s['envelope_us']-s['forward_us'] for s in samples))
       assert all(s['selected_tiles']==samples[0]['selected_tiles'] for s in samples)
       for rank,s in enumerate(samples):
        assert s['correctness'];checks+=1
        origin,end=s['local_markers'];assert end>origin>0
        groups=defaultdict(list)
        for r in s.get('producer',[]):groups[r['tile']].append(r)
        for tile,fs in groups.items():
         assert {r['fragment'] for r in fs}==set(range(rows[rank]['fragment_count']))
         assert all(r['values'][7]==1 and r['values'][1]>=r['values'][0]>0 for r in fs)
         producers.append(dict(ident,mode=mode,repeat=repeat,source_rank=rank,tile=tile,target_rank=fs[0]['values'][5],
            ready_after_own_origin_us=(max(r['values'][0] for r in fs)-origin)/1000,
            stores_end_after_own_origin_us=(max(r['values'][1] for r in fs)-origin)/1000,
            end_semantics='issued_by_all_threads' if mode=='producer_sparse' else 'system_fenced_before_release'))
        if 'receiver' not in s:continue
        st=s['observer_status'];assert st[2]==0 and st[3]==len(s['receiver'])
        censored=sum(r['observed_ns']<0 for r in s['receiver'])
        validity.append(dict(ident,mode=mode,repeat=repeat,target_rank=rank,fragments=len(s['receiver']),first_scan_visible=censored,poll_cycle_us=s['max_poll_cycle_ns']/1000))
        groups=defaultdict(list)
        for r in s['receiver']:groups[r['tile']].append(r)
        per=rows[rank]['partition_tiles']
        for tile,fs in groups.items():
         assert tile//per==rank
         joined=join_observation(fs,world,rows[rank]['fragment_count'],s['max_poll_cycle_ns'],origin,end)
         if joined is None:continue
         receivers.append(dict(ident,mode=mode,repeat=repeat,target_rank=rank,tile=tile,offset=tile%per,
                               is_partition_last_tile=tile%per==per-1,**joined))
      median=med(overheads);noisy=cv(off_times)>5 or q(overheads,.75)-q(overheads,.25)>10
      perf.append(dict(ident,mode=mode,median_forward_us=med(by_rep),median_delta_us=med(deltas),median_overhead_percent=median,
                       overhead_p25=q(overheads,.25),overhead_p75=q(overheads,.75),off_cv_percent=cv(off_times),
                       max_observer_extra_us=max(extras),screening='HIGH_PERTURBATION' if abs(median)>5 else 'NOISY' if noisy else 'LOW_MEDIAN_PERTURBATION'))
 assert len(set(identities.values()))==1
 # Actual coordinates, epochs and conditions must match across the three policies.
 for block,order in enumerate(orders):
  for si in range(len(cfg['shapes'])):
   reference=data[(block,order[0],'instrumented')][0]['shapes'][si]['samples']
   ref={(s['repeat'],s['mode']):s['selected_tiles'] for s in reference}
   for policy in order:
    samples=data[(block,policy,'instrumented')][0]['shapes'][si]['samples']
    assert {(s['repeat'],s['mode']):s['selected_tiles'] for s in samples}==ref
 controls=[]
 for block,order in enumerate(orders):
  for policy in order:
   for m,n,k in cfg['shapes']:
    group={r['kind']:r for r in builds if (r['block'],r['policy'],r['M'],r['N'],r['K_global'])==(block,policy,m,n,k)}
    assert len({(r['hparams'],r['grid'],r['threads']) for r in group.values()})==1
    before,after,off=[group[x]['median_us'] for x in ('vanilla_before','vanilla_after','instrumented')]
    controls.append(dict(block=block,policy=policy,M=m,N=n,K_global=k,
        instrumented_off_overhead_percent=100*(off/math.sqrt(before*after)-1),
        vanilla_drift_percent=100*(after/before-1),vanilla_registers=group['vanilla_before']['registers'],sample_registers=group['instrumented']['registers']))
 joins=[]
 keys=sorted({(r['block'],r['policy'],r['M'],r['N'],r['K_global'],r['mode']) for r in receivers})
 for block,policy,m,n,k,mode in keys:
  rs=[r for r in receivers if (r['block'],r['policy'],r['M'],r['N'],r['K_global'],r['mode'])==(block,policy,m,n,k,mode)]
  resolved=[r for r in rs if r['last_source_resolved']];late=[r for r in rs if r['is_partition_last_tile']]
  joins.append(dict(block=block,policy=policy,M=m,N=n,K_global=k,mode=mode,observations=len(rs),
       resolved=len(resolved),remote_last=sum(r['last_source']!=r['target_rank'] for r in resolved),
       skew_p50_us=med([r['join_skew_us'] for r in rs]),skew_p95_us=q([r['join_skew_us'] for r in rs],.95),
       arrival_p50_us=med([r['last_after_origin_us'] for r in rs]),arrival_p95_us=q([r['last_after_origin_us'] for r in rs],.95),
       last_tile_arrival_p50_us=med([r['last_after_origin_us'] for r in late]),last_tile_end_gap_p50_us=med([r['end_minus_last_us'] for r in late]),
       max_poll_cycle_us=max(r['poll_cycle_us'] for r in rs)))
 comparisons=[]
 indexed={(r['block'],r['policy'],r['M'],r['N'],r['K_global'],r['mode'],r['repeat'],r['target_rank'],r['tile']):r for r in receivers}
 for block,order in enumerate(orders):
  for m,n,k in cfg['shapes']:
   for policy in ('remote_first','interleaved'):
    for late_only in (False,True):
     pairs=[]
     for r in receivers:
      if (r['block'],r['policy'],r['M'],r['N'],r['K_global'],r['mode'])!=(block,policy,m,n,k,'receiver_sparse'):continue
      if late_only and not r['is_partition_last_tile']:continue
      key=(block,'original',m,n,k,'receiver_sparse',r['repeat'],r['target_rank'],r['tile'])
      if key in indexed:pairs.append((indexed[key],r))
     comparisons.append(dict(block=block,policy=policy,M=m,N=n,K_global=k,last_tiles_only=late_only,pairs=len(pairs),
         arrival_delta_us=med([b['last_after_origin_us']-a['last_after_origin_us'] for a,b in pairs]),
         skew_delta_us=med([b['join_skew_us']-a['join_skew_us'] for a,b in pairs]),
         operator_span_delta_us=med([b['forward_span_us']-a['forward_span_us'] for a,b in pairs]),
         poll_uncertainty_p95_us=q([2*max(a['poll_cycle_us'],b['poll_cycle_us']) for a,b in pairs],.95)))
 for name,rows in [('builds.csv',builds),('build_controls.csv',controls),('sampling_overhead.csv',perf),('receiver_tiles.csv',receivers),('producer_tiles.csv',producers),('validity.csv',validity),('join_summary.csv',joins),('arrival_policy_pairs.csv',comparisons)]:write(out,name,rows)
 admission=[]
 for control in controls:
  key=tuple(control[k] for k in ('block','policy','M','N','K_global'))
  sample=next(r for r in perf if tuple(r[k] for k in ('block','policy','M','N','K_global'))==key and r['mode']=='receiver_sparse')
  admission.append(dict(**{k:control[k] for k in ('block','policy','M','N','K_global')},
       build_off_overhead_percent=control['instrumented_off_overhead_percent'],
       sampling_overhead_percent=sample['median_overhead_percent'],off_cv_percent=sample['off_cv_percent'],
       accepted=abs(control['instrumented_off_overhead_percent'])<=5 and abs(control['vanilla_drift_percent'])<=5 and sample['screening']=='LOW_MEDIAN_PERTURBATION'))
 write(out,'measurement_admission.csv',admission)
 report=dict(sample_forward_checks=checks,first_scan_visible=sum(r['first_scan_visible'] for r in validity),
             low_perturbation_windows=sum(r['screening']=='LOW_MEDIAN_PERTURBATION' for r in perf if r['mode']=='receiver_sparse'),
             sparse_windows=sum(r['mode']=='receiver_sparse' for r in perf),accepted_measurement_windows=sum(r['accepted'] for r in admission),smoke=smoke,
             limitations=['Release observations include system fence, flag propagation and polling.',
              'Four-source last-rank resolution uses the latest-versus-second-latest gap, not total skew.',
              'Only sampled output coordinates; coarse GPU forward boundary, no per-tile reduction consumption stamp.',
              'Single-forward diagnostics include common marker overhead and host submission gaps.',
              'Cross-policy arrival deltas use matched receiver/tile/repeat and local receiver origins; rank launch skew can still contribute.',
              'Nsight hardware counters require permission; timing alone cannot establish cache causality.'])
 (out/'analysis.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2));return report

if __name__=='__main__':analyze(Path(sys.argv[1]))
