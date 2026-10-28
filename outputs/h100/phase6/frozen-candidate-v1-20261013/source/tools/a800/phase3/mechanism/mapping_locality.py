"""Logical input-band reuse, not hardware cache hit rate or actual CTA execution order."""
import csv,statistics,sys
from pathlib import Path
source=Path(sys.argv[1]);out=Path(sys.argv[2]);rows=[]
for m,n,k in [(8192,4096,2048),(4096,4096,8192),(4096,4160,8192)]:
 for policy in ['original','remote_first','interleaved']:
  path=source/f'tp4-m{m}-n{n}-k{k}'/f'mapping-{policy}.csv'
  records=[{key:int(value) for key,value in r.items()} for r in csv.DictReader(path.open())]
  for rank in range(4):
   rs=sorted([r for r in records if r['rank']==rank],key=lambda r:r['index'])
   pairs=list(zip(rs,rs[1:]));a=[];b=[];footprints=[]
   for i in range(0,len(rs),32):
    group=rs[i:i+32];ams={r['m'] for r in group};bns={r['n'] for r in group}
    a.append(len(ams));b.append(len(bns))
    footprints.append((len(ams)*128+sum(min(128,n-x*128) for x in bns))*(k//4)*2)
   rows.append(dict(M=m,N=n,K_global=k,policy=policy,rank=rank,tiles=len(rs),
     consecutive_same_A_band_fraction=sum(x['m']==y['m'] for x,y in pairs)/len(pairs),
     consecutive_same_B_band_fraction=sum(x['n']==y['n'] for x,y in pairs)/len(pairs),
     mean_A_bands_per_32_logical_tiles=statistics.mean(a),mean_B_bands_per_32_logical_tiles=statistics.mean(b),
     mean_unique_input_bytes_per_32_logical_tiles=statistics.mean(footprints),
     semantics='Logical mapping only; not physical scheduling, actual memory traffic, or cache misses'))
with out.open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print('Saved',len(rows),'logical mapping diagnostics to',out)
