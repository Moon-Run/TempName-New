"""Aggregate bounded tile samples; never subtract different GPUs' timestamps."""
import csv,json,re,statistics,sys
from collections import defaultdict,Counter
from pathlib import Path
out=Path(sys.argv[1])
kinds=['original_before','instrumented','original_after']
data={kind:[json.loads((out/f'{kind}-rank{r}.json').read_text()) for r in range(2)] for kind in kinds}
assert all(d['completed'] for ranks in data.values() for d in ranks)
def med(x):return statistics.median(x)
def quantile(x,p):
 s=sorted(x);i=(len(s)-1)*p;lo=int(i);hi=min(lo+1,len(s)-1);return s[lo]+(s[hi]-s[lo])*(i-lo)
def csvwrite(name,rows):
 with (out/name).open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def entry(rank,ki,mode,repeat):
 return next(s for s in data['instrumented'][rank]['shapes'][ki]['samples'] if s['mode']==mode and s['repeat']==repeat)
perf=[];builds=[];joins=[];producers=[];validity=[];lastcounts=[]
for ki,k in enumerate([2048,8192]):
 identities=set()
 for kind,ranks in data.items():
  samples=[max(d['shapes'][ki]['batch_us'][i] for d in ranks) for i in range(10)]
  builds.append(dict(K_global=k,build=kind,median_us=med(samples),min_us=min(samples),max_us=max(samples),
                     cv_percent=100*statistics.stdev(samples)/statistics.mean(samples)))
  for d in ranks:
   name=d['shapes'][ki]['kernels'][0]
   match=re.search(r'(64x64x32_16x8x16_streamksk_nil_\d+x\d+x\d+_gemmstreamk_\d+_rasterheuristic)',name)
   assert match,name
   identities.add(match.group(1))
 assert len(identities)==1,'GEMM hparams changed across builds/ranks'
 for mode in ['off','producer_sparse','publication_sparse','receiver_sparse','receiver_dense']:
  forward=[];envelope=[];overhead=[];extra=[]
  for repeat in range(20):
   samples=[entry(r,ki,mode,repeat) for r in range(2)]
   control=[entry(r,ki,'off',repeat) for r in range(2)]
   v=max(s['forward_us'] for s in samples);base=max(s['forward_us'] for s in control)
   forward.append(v);envelope.append(max(s['envelope_us'] for s in samples));overhead.append(100*(v/base-1));extra.append(v-base)
  perf.append(dict(K_global=k,mode=mode,median_forward_us=med(forward),median_envelope_us=med(envelope),
     median_paired_delta_us=med(extra),median_paired_overhead_percent=med(overhead),
     overhead_p25=quantile(overhead,.25),overhead_p75=quantile(overhead,.75),
     overhead_p95=quantile(overhead,.95),screening='PASS_MEDIAN_5_PERCENT' if med(overhead)<=5 else 'HIGH_PERTURBATION',
     hparams=next(iter(identities))))
  for rank in range(2):
   for repeat in range(20):
    s=entry(rank,ki,mode,repeat);assert s['correctness']
    ps=defaultdict(list)
    for r in s.get('producer',[]):
     v=r['values'];assert v[7]==1 and v[1]>=v[0]>0
     ps[r['tile']].append(r)
    for tile,frags in ps.items():
     expected=frags[0]['values'][4]
     assert {f['fragment'] for f in frags}==set(range(expected))
     ready=max(r['values'][0] for r in frags);end=max(r['values'][1] for r in frags)
     producers.append(dict(K_global=k,mode=mode,repeat=repeat,source_rank=rank,target_rank=frags[0]['values'][5],
        tile=tile,fragments=expected,ready_after_local_origin_us=(ready-s['local_markers'][0])/1000,
        stores_end_after_local_origin_us=(end-s['local_markers'][0])/1000,
        full_ready_to_end_us=(end-ready)/1000,
        end_semantics='issued_by_all_threads' if mode=='producer_sparse' else 'system_fenced_before_release'))
    if 'receiver' not in s:continue
    st=s['observer_status'];assert st[2]==0 and st[3]==len(s['receiver'])
    groups=defaultdict(lambda:defaultdict(list))
    for r in s['receiver']:groups[r['tile']][r['source']].append(r['observed_ns'])
    censored=sum(r['observed_ns']<0 for r in s['receiver'])
    validity.append(dict(K_global=k,mode=mode,rank=rank,repeat=repeat,observed_fragments=len(s['receiver']),
          first_scan_visible=censored,max_poll_cycle_us=s['max_poll_cycle_ns']/1000,timeout=st[2]))
    for tile,sources in groups.items():
     assert set(sources)=={0,1}
     assert all(len(v)==data['instrumented'][rank]['shapes'][ki]['fragment_count'] for v in sources.values())
     if any(t<0 for ts in sources.values() for t in ts):continue
     times={src:max(ts) for src,ts in sources.items()}
     diff=abs(times[0]-times[1]);latest=max(times,key=times.get)
     # Allow one polling cycle of uncertainty at either endpoint.
     robust=diff>2*s['max_poll_cycle_ns']
     joins.append(dict(K_global=k,mode=mode,repeat=repeat,target_rank=rank,tile=tile,
        source0_all_fragments_observed_ns=times[0],source1_all_fragments_observed_ns=times[1],
        last_source=latest,last_is_remote=latest!=rank,observed_join_skew_us=diff/1000,
        all_observed_after_observer_start_us=(max(times.values())-st[0])/1000,
        max_poll_cycle_us=s['max_poll_cycle_ns']/1000,order_resolved_above_poll_cycle=robust))
 for mode in ['receiver_sparse','receiver_dense']:
  group=[j for j in joins if j['K_global']==k and j['mode']==mode]
  robust=[j for j in group if j['order_resolved_above_poll_cycle']]
  per_tile=defaultdict(list)
  for j in robust:per_tile[(j['target_rank'],j['tile'])].append(j['last_source'])
  lastcounts.append(dict(K_global=k,mode=mode,tile_observations=len(group),unique_target_tiles=len(per_tile),
      resolved_observations=len(robust),remote_last=sum(j['last_is_remote'] for j in robust),
      median_observed_skew_us=med([j['observed_join_skew_us'] for j in group]),
      p95_observed_skew_us=quantile([j['observed_join_skew_us'] for j in group],.95),
      minimum_skew_us=min(j['observed_join_skew_us'] for j in group),
      max_poll_cycle_us=max(j['max_poll_cycle_us'] for j in group),
      repeated_tiles=len([v for v in per_tile.values() if len(v)>=2]),
      consistent_repeated_tiles=sum(len(set(v))==1 for v in per_tile.values() if len(v)>=2)))
csvwrite('performance.csv',perf);csvwrite('build_controls.csv',builds);csvwrite('producer_tiles.csv',producers)
csvwrite('receiver_tiles.csv',joins);csvwrite('sampling_validity.csv',validity);csvwrite('join_summary.csv',lastcounts)
summary=dict(performance=perf,build_controls=builds,join_summary=lastcounts,
    measured_forward_checks=sum(len(d['shapes'][i]['samples']) for d in data['instrumented'] for i in range(2)),
    first_scan_visible=sum(v['first_scan_visible'] for v in validity),
    clock_checks=[d['clock_check'] for d in data['instrumented']],
    limitations=['Observed release-marker times include producer fencing, marker propagation, and polling.',
                 'No cross-GPU timestamp subtraction. No per-tile final-consumption timestamp.',
                 'GPU-visible end marker is a coarse post-forward boundary including host enqueue gaps.',
                 'A sampled remote-last pattern does not prove that changing order improves full-op latency.',
                 'Sampler-off versus original-build comparisons are separate process windows, not simultaneous pairs.'])
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps({'build_controls':builds,'join_summary':lastcounts},indent=2))
