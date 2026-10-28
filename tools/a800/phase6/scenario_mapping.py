"""Audit actual GPU attention coordinates and each model's calibrated MLP table."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1]);info=json.loads((root/'submission.json').read_text())
config=json.loads((root/'scripts/config.json').read_text());case=config['cases'][0]
plan=json.loads((root/'scripts/selection-plan.json').read_text())
build=Path(info['artifact_root'])/'build';out=root/'results/mapping';checks=[]
m=case['sequence']*config['common']['micro_batch'];n=case['hidden'];tiles=m//128*(n//128)
for policy in info['mappings']:
    for k,mlp in ((case['hidden'],False),(case['ffn'],True)):
        run=subprocess.run([str(build/'mapping'/policy),str(m),str(n),str(k),'4',str(int(mlp))],capture_output=True,text=True,check=True)
        (out/f'{policy}-k{k}.csv').write_text(run.stdout);(out/f'{policy}-k{k}.log').write_text(run.stderr)
        rows=list(csv.DictReader(io.StringIO(run.stdout)));assert len(rows)==4*tiles
        if mlp:
            base='remote_first' if policy=='remote_arrival' else policy.removesuffix('_arrival')
            entry=next(s for s in plan['policies'][base] if s['shape']==[m,n,k])
            for r in rows:
                rank,idx=int(r['rank']),int(r['index'])
                assert int(r['m'])*(n//128)+int(r['n'])==entry['coords'][rank][idx]
        checks.append(dict(policy=policy,shape=[m,n,k],mlp=mlp,passed=True))
(out/'verification.json').write_text(json.dumps(dict(completed=True,checks=checks),indent=2)+'\n')
