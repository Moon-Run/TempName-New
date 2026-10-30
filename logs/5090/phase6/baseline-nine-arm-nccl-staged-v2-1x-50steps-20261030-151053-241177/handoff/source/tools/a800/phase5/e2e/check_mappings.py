"""GPU mapping audit: attention identity, MLP exactly matches calibration order."""
import csv
import io
import json
from pathlib import Path
import subprocess
import sys

root=Path(sys.argv[1]);info=json.loads((root/'submission.json').read_text())
build=Path(info['artifact_root'])/'build'
plan=json.loads((root/'scripts/selection-plan.json').read_text())
out=root/'results/mapping';results=[]
for policy in info.get('mappings',['remote_first','interleaved']):
    pid={'remote_first':1,'interleaved':2,'remote_arrival':3,'interleaved_arrival':4}[policy]
    for k,expected in ((2048,0),(8192,pid)):
        command=[str(build/'mapping'/policy),'2048','2048',str(k),'4',str(expected)]
        run=subprocess.run(command,text=True,capture_output=True,check=True)
        (out/f'{policy}-k{k}.csv').write_text(run.stdout)
        (out/f'{policy}-k{k}.log').write_text(run.stderr)
        rows=list(csv.DictReader(io.StringIO(run.stdout)))
        assert len(rows)==1024
        if k==8192:
            base={'remote_arrival':'remote_first','interleaved_arrival':'interleaved'}.get(policy,policy)
            source=next(Path(p) for p in plan['inputs'] if Path(p).name==f'mapping-{base}-2048-2048-8192.csv')
            reference=list(csv.DictReader(source.open()))
            fields=('rank','index','m','n','destination')
            if policy.endswith('_arrival'):
                ordering=json.loads((root/'scripts/arrival-plan.json').read_text())
                shape=next(s for s in ordering['policies'][policy]['shapes'] if (s['M'],s['N'],s['K_global'])==(2048,2048,8192))
                for r,old in zip(rows,reference):
                    rank,idx=int(r['rank']),int(r['index'])
                    mapped=shape['maps'][rank][idx]
                    expected=reference[rank*256+mapped]
                    assert int(r['m'])*16+int(r['n'])==shape['coords'][rank][idx]
                    assert (r['m'],r['n'],r['destination'])==(expected['m'],expected['n'],expected['destination'])
                    assert r['destination']==old['destination']
                    slots=[int(x['index']) for x in reference[rank*256:(rank+1)*256] if x['destination']==old['destination']]
                    assert slots.index(idx)//ordering['window']==slots.index(mapped)//ordering['window']
                # A selection bit belongs to a physical tile, never to its new slot.
                mask=plan['policies'][base]['mask']
                for rank in range(4):
                    old_set={int(x['m'])*16+int(x['n']) for x in reference[rank*256:(rank+1)*256] if mask[rank][int(x['m'])*16+int(x['n'])]}
                    new_set={int(x['m'])*16+int(x['n']) for x in rows[rank*256:(rank+1)*256] if mask[rank][int(x['m'])*16+int(x['n'])]}
                    assert old_set==new_set
            else:
                assert [tuple(r[f] for f in fields) for r in rows]==[tuple(r[f] for f in fields) for r in reference]
        results.append(dict(policy=policy,K_global=k,passed=True,base_or_arrival_plan_match=k==8192))
(out/'verification.json').write_text(json.dumps(dict(completed=True,checks=results),indent=2)+'\n')
