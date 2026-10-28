"""Per-shape, per-base arrival permutations and bounded physical selection masks."""
import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('arrival_plan',HERE.parent/'arrival/plan.py')
arrival=importlib.util.module_from_spec(spec);spec.loader.exec_module(arrival)
spec=importlib.util.spec_from_file_location('selection_plan',HERE.parent/'phase5/plan.py')
selection=importlib.util.module_from_spec(spec);spec.loader.exec_module(selection)


def main(source,out,fraction=1/64,window=64):
    cfg=json.loads((source/'calibration-complete.json').read_text());assert cfg['completed']
    assert cfg['config']['world_size']==4, 'Use tp8_quant/plan.py for eight-rank calibration'
    info=json.loads((source/'plan.json').read_text())
    result=dict(schema=2,world=4,window=window,budget_fraction=fraction,training_passes=[0,1],heldout_passes=[2],
                shape_catalog=cfg['config']['shapes'],policies={},inputs={},
                limitation='Per-base BF16 calibration, including the new R1,L,R2,R3 base. Not post-quantization fixed-point recalibration.')
    for policy in info['policies']:
        paths=[source/f'b0-{policy}-instrumented-rank{r}.json' for r in range(4)]
        reports=[json.loads(p.read_text()) for p in paths]
        assert all(r['completed'] for r in reports)
        entries=[]
        for m,n,k in result['shape_catalog']:
            tiles=m//128*((n+127)//128);per=tiles//4;nt=(n+127)//128
            scores=[[0.]*tiles for _ in range(4)];heldout={};overhead=[];invalid=[];training_valid=0
            for target,report in enumerate(reports):
                sh=next(s for s in report['shapes'] if (s['M'],s['N'],s['K_global'])==(m,n,k))
                obs=arrival.observations(sh,target,4,relative=True)
                for tile in range(target*per,(target+1)*per):
                    invalid += [dict(receiver=target,pass_index=p,tile=tile) for p in (0,1,2) if (p,tile) not in obs]
                    if all((p,tile) in obs for p in (0,1)):
                        training_valid+=1
                        values=arrival.priority([obs[(p,tile)] for p in (0,1)],4,'tail')
                        for src in range(4):scores[src][tile]=values[src]
                    # Invalid training observations give zero priority/BF16.
                    # Held-out availability never gates fitting or selection.
                    if (2,tile) in obs:heldout[tile]=obs[(2,tile)]
                off={s['repeat']:s['forward_us'] for s in sh['samples'] if s['mode']=='off'}
                overhead += [100*(s['forward_us']/off[s['repeat']]-1) for s in sh['samples'] if s['mode']=='receiver_sparse']
            mapping_path=source/f'mapping-{policy}-{m}-{n}-{k}.csv';paths.append(mapping_path)
            rows=[{key:int(v) for key,v in r.items()} for r in csv.DictReader(mapping_path.open())]
            assert len(rows)==tiles*4
            maps=[];coords=[];base_coords=[];changed=[]
            for rank in range(4):
                rr=sorted([dict(r,tn=nt) for r in rows if r['rank']==rank],key=lambda r:r['index'])
                assert [r['index'] for r in rr]==list(range(tiles))
                baseline=[r['m']*nt+r['n'] for r in rr]
                assert sorted(baseline)==list(range(tiles))
                for i,r in enumerate(rr):
                    if policy.startswith('interleaved'):
                        slot=i%4
                        if policy=='interleaved_remote' and slot<2:slot^=1
                        if policy=='interleaved_remote_group':slot=(slot+1)%4
                        dst=(slot+rank)%4;within=i//4
                        assert (r['m'],r['n'])==(dst*(m//128//4)+within//nt,within%nt)
                    else:assert (r['m'],r['n'])==((r['base_m']+m//128//4*(rank+1))%(m//128),r['base_n'])
                mapping=arrival.permute(rr,{t:v for t,v in enumerate(scores[rank])},window)
                assert sorted(mapping)==list(range(tiles))
                slots={d:[i for i,r in enumerate(rr) if r['destination']==d] for d in range(4)}
                position={i:(d,j//window) for d,indices in slots.items() for j,i in enumerate(indices)}
                assert all(position[i]==position[j] for i,j in enumerate(mapping))
                maps.append(mapping);coords.append([baseline[i] for i in mapping]);base_coords.append(baseline)
                changed.append(sum(i!=j for i,j in enumerate(mapping)))
            mask=selection.select(scores,4,fraction)
            chosen=[(s,t) for s in range(4) for t in range(tiles) if mask[s][t]]
            hits=[max(heldout[t][0])-heldout[t][0][s]<=heldout[t][1] for s,t in chosen if t in heldout]
            entries.append(dict(shape=[m,n,k],M=m,N=n,K_global=k,K_local=k//4,tiles=tiles,
                                maps=maps,coords=coords,base_coords=base_coords,changed_per_rank=changed,
                                mask=mask,selected_per_source=[sum(r) for r in mask],
                                training_valid_tiles=training_valid,invalid_observations=invalid,
                                heldout_selected_observations=len(hits),
                                selected_remote_fraction=len(chosen)/(3*tiles),
                                heldout_near_critical_fraction=statistics.mean(hits) if hits else None,
                                sampled_median_overhead_percent=statistics.median(overhead)))
            print(policy,m,n,k,'changed',changed,'selected',entries[-1]['selected_per_source'],flush=True)
        result['policies'][policy]=entries
        for p in paths:result['inputs'][str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    for p in (source/'calibration-complete.json',source/'plan.json',Path(__file__),HERE.parent/'arrival/plan.py',HERE.parent/'phase5/plan.py'):
        result['inputs'][str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists()
    out.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    p.add_argument('--fraction',type=float,default=1/64);p.add_argument('--window',type=int,default=64)
    a=p.parse_args();main(a.source,a.out,a.fraction,a.window)
