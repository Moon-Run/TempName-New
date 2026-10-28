"""Read-only replay: label ranking inside the actual legal 16-slot windows."""
import csv,hashlib,json,statistics,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[3]
OUT=REPO/'logs/a800/arrival/arrival-v2-tp4-analysis-20261003'
OUT.mkdir(parents=True, exist_ok=True)
ROOT=REPO/'outputs/a800/arrival-v2-tp4-20261003'
sys.path.insert(0,str(ROOT/'source/tools/a800/arrival'))
from plan import observations
from validate_calibration import spearman
TRAIN=ROOT/'results/calibration-fit'
VALID=ROOT/'results/calibration-validation'
inputs={}
def read(p):
 raw=p.read_bytes();inputs[str(p)]=hashlib.sha256(raw).hexdigest();return json.loads(raw)
records=[]
for policy in ['remote_first','interleaved']:
 tr=[read(TRAIN/f'b0-{policy}-instrumented-rank{r}.json') for r in range(4)]
 va=[read(VALID/f'b0-{policy}-instrumented-rank{r}.json') for r in range(4)]
 for si in range(4):
  sh=tr[0]['shapes'][si];m,n,k=[sh[x] for x in ('M','N','K_global')];tn=n//128;per=sh['partition_tiles'];scores={}
  for target in range(4):
   a=observations(tr[target]['shapes'][si],target,4,True);b=observations(va[target]['shapes'][si],target,4,True)
   for t in range(target*per,(target+1)*per):
    if all((p,t) in a for p in (0,1)) and all((p,t) in b for p in (0,1,2)):
     scores[t]=(statistics.median(max(a[p,t][0]) for p in (0,1)),statistics.median(max(b[p,t][0]) for p in (0,1,2)))
  mapping=TRAIN/f'mapping-{policy}-{m}-{n}-{k}.csv'
  inputs[str(mapping)]=hashlib.sha256(mapping.read_bytes()).hexdigest()
  rows=[{key:int(v) for key,v in r.items()} for r in csv.DictReader(mapping.open())]
  groups=set()
  for rank in range(4):
   rr=sorted([r for r in rows if r['rank']==rank],key=lambda r:r['index'])
   for dest in range(4):
    ts=[r['m']*tn+r['n'] for r in rr if r['destination']==dest]
    for start in range(0,len(ts),16):groups.add(tuple(sorted(ts[start:start+16])))
  windows=[]
  for group in sorted(groups):
   ts=[t for t in group if t in scores]
   if len(ts)<8:continue
   rho=spearman([scores[t][0] for t in ts],[scores[t][1] for t in ts])
   if rho is not None:windows.append(dict(tiles=ts,spearman=rho))
  cs=[w['spearman'] for w in windows]
  records.append(dict(policy=policy,M=m,N=n,K_global=k,unique_legal_windows=len(groups),evaluated_windows=len(cs),within_window_spearman_median=statistics.median(cs),within_window_spearman_min=min(cs),within_window_spearman_max=max(cs),windows=windows))
result=dict(scope='Offline replay of existing base-order training and separate validation samples; not new GPU or post-reorder evidence. Each unique destination-preserving 16-slot physical-tile set counted once across source ranks. Uses all paired valid tiles in each window (at least 8), not just source-admitted candidates; evaluates join ordering, not causal speedup.',training=str(TRAIN),validation=str(VALID),inputs=inputs,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=records)
(OUT/'window-audit.json').write_text(json.dumps(result,indent=2)+'\n')
for r in records:print(r['policy'],r['M'],r['N'],r['K_global'],r['evaluated_windows'],round(r['within_window_spearman_median'],3))
