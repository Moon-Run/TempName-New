"""Fit H100-only arrival and selective masks using training passes 0/1."""
import argparse
import csv
from pathlib import Path
import statistics
from common import HERE, REPO, PROTOCOL, config, load, read, require, sha, shape, validate_entry, verify_files, write

arrival=load('h100_arrival','tools/a800/arrival/plan.py')
selection=load('h100_selection','tools/a800/phase5/plan.py')


def validate_mapping(rows,c,base):
    m,n,_=shape(c);world=c['tp'];tm,tn=m//128,n//128;tiles=tm*tn
    require(len(rows)==tiles*world,'Missing mapping rows')
    require({r['rank'] for r in rows}==set(range(world)),'Missing mapping ranks')
    for rank in range(world):
        rr=sorted((r for r in rows if r['rank']==rank),key=lambda r:r['index'])
        require([r['index'] for r in rr]==list(range(tiles)),'Duplicate/missing logical slots')
        require(sorted(r['m']*tn+r['n'] for r in rr)==list(range(tiles)),'Non-bijective base map')
        for r in rr:
            i=r['index']
            if base=='remote_first':
                expected=((r['base_m']+tm//world*(rank+1))%tm,r['base_n'])
            else:
                slot=i%world
                if base=='interleaved_remote' and slot<2:slot^=1
                if base=='interleaved_remote_group':slot=(slot+1)%world
                dst=(slot+rank)%world;within=i//world
                expected=(dst*(tm//world)+within//tn,within%tn)
            require((r['m'],r['n'])==expected and r['destination']==r['m']//(tm//world),'Base order disagrees with calibration mapping')


def fit(reports,rows,c,base):
    """Pure CPU fit; held-out observations are used only for diagnostics."""
    validate_mapping(rows,c,base)
    m,n,k=shape(c);world=c['tp'];tiles=m//128*(n//128);per=tiles//world
    scores=[[0.]*tiles for _ in range(world)];heldout={};overhead=[];invalid=[];valid=0
    require(len(reports)==world,'Missing calibration rank')
    for rank,r in enumerate(reports):
        require(r['rank']==rank and r['completed'],'Wrong/incomplete calibration rank')
        sh=next(s for s in r['shapes'] if [s['M'],s['N'],s['K_global']]==[m,n,k])
        obs=arrival.observations(sh,rank,world,relative=True)
        for t in range(rank*per,(rank+1)*per):
            invalid.extend(dict(rank=rank,pass_index=p,tile=t) for p in range(3) if (p,t) not in obs)
            if all((p,t) in obs for p in (0,1)):
                valid+=1
                for src,value in enumerate(arrival.priority([obs[(p,t)] for p in (0,1)],world,'tail')):
                    scores[src][t]=value
            if (2,t) in obs:heldout[t]=obs[(2,t)]
        off={s['repeat']:s['forward_us'] for s in sh['samples'] if s['mode']=='off'}
        overhead += [100*(s['forward_us']/off[s['repeat']]-1) for s in sh['samples'] if s['mode']=='receiver_sparse']
    maps=[];coords=[];base_coords=[]
    for rank in range(world):
        rr=sorted([dict(r,tn=n//128) for r in rows if r['rank']==rank],key=lambda r:r['index'])
        original=[r['m']*(n//128)+r['n'] for r in rr]
        mapping=arrival.permute(rr,dict(enumerate(scores[rank])),c['window'])
        maps.append(mapping);coords.append([original[i] for i in mapping]);base_coords.append(original)
    mask=selection.select(scores,world,c['fraction'])
    hits=[max(heldout[t][0])-heldout[t][0][s]<=heldout[t][1]
          for s in range(world) for t in range(tiles) if mask[s][t] and t in heldout]
    entry=dict(shape=[m,n,k],M=m,N=n,K_global=k,K_local=k//world,tiles=tiles,maps=maps,coords=coords,
               base_coords=base_coords,mask=mask,selected_per_source=list(map(sum,mask)),
               changed_per_rank=[sum(i!=j for i,j in enumerate(row)) for row in maps],
               training_valid_tiles=valid,invalid_observations=invalid,heldout_selected_observations=len(hits),
               heldout_near_critical_fraction=statistics.mean(hits) if hits else None,
               sampled_median_overhead_percent=statistics.median(overhead) if overhead else None)
    validate_entry(entry,c)
    return entry


def main(source,out):
    source,out=source.resolve(),out.resolve();done=read(source/'calibration-complete.json')
    require(done['protocol']==PROTOCOL and done['arch']=='sm90' and done['completed'] and done['gpu_calibrated'],'H100 GPU calibration required')
    c=done['config'];world=c['tp'];m,n,k=shape(c)
    require(not out.exists(),'Do not overwrite a frozen plan')
    preparation=read(source/'plan.json');verify_files(source,preparation['files'])
    require(done['run_id']==preparation['run_id'] and done['config']==preparation['config'],'Calibration provenance mismatch')
    require(sha(Path(done['build'])/'manifest.json')==done['build_sha256'],'Calibration build changed')
    result=dict(protocol=PROTOCOL,schema=2,arch='sm90',world=world,config=c,gpu_calibrated=True,
                window=c['window'],budget_fraction=c['fraction'],shape_catalog=[shape(c)],policies={},inputs={},
                hardware=done['hardware'],training_passes=[0,1],heldout_passes=[2],
                limitation='H100 BF16 per-base calibration; not recalibrated after ordering/quantization; sampling overhead is diagnostic.')
    files=[source/'calibration-complete.json',source/'plan.json',HERE/'plan.py',HERE/'common.py',
           REPO/'tools/a800/arrival/plan.py',REPO/'tools/a800/phase5/plan.py']
    for base in c['bases']:
        paths=[source/f'b0-{base}-instrumented-rank{r}.json' for r in range(world)]
        reports=[read(p) for p in paths]
        for rank,r in enumerate(reports):
            require(r['run_id']==done['run_id'] and r['policy']==base and r['world_size']==world and not r['smoke'],'Calibration run mismatch')
            require(r['hardware']['devices']==done['hardware']['devices'] and r['hardware']['host']==done['hardware']['host'],'Mixed calibration hardware')
            require(r['hardware']['local_rank']==rank,'Not a complete local TP group')
        mapping=source/f'mapping-{base}-{m}-{n}-{k}.csv'
        with mapping.open() as f:rows=[{k:int(v) for k,v in row.items()} for row in csv.DictReader(f)]
        result['policies'][base]=[fit(reports,rows,c,base)]
        files+=paths+[mapping]
    result['inputs']={str(p):sha(p) for p in files}
    out.parent.mkdir(parents=True,exist_ok=True);write(out,result)
    print(out)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('source',type=Path);p.add_argument('out',type=Path)
    a=p.parse_args();main(a.source,a.out)
