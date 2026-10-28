"""Fit bounded arrival-priority permutations on passes 0/1; pass 2 is held out.

Default: prioritize late joins, measured from each invocation's receiver-local
start marker, for sources consistently near-critical in both training passes.
The gap ablation requires the same resolved last source in both passes.
This is an exploratory heuristic, not a calibrated network-time model.
Every destination slot and per-destination window is preserved.
"""
import argparse,csv,hashlib,json,math,statistics
from pathlib import Path

def permute(rows,scores,window):
    assert window>0
    rows=sorted(rows,key=lambda r:r['index'])
    assert [r['index'] for r in rows]==list(range(len(rows)))
    mapping=list(range(len(rows)))
    for dest in sorted({r['destination'] for r in rows}):
        slots=[r['index'] for r in rows if r['destination']==dest]
        for start in range(0,len(slots),window):
            group=slots[start:start+window]
            ordered=sorted(group,key=lambda i:-scores.get(rows[i]['m']*rows[i]['tn']+rows[i]['n'],0))
            for i,j in zip(group,ordered):mapping[i]=j
    assert sorted(mapping)==list(range(len(rows)))
    assert all(rows[i]['destination']==rows[j]['destination'] for i,j in enumerate(mapping))
    return mapping

def observations(shape,rank,world=4,relative=False):
    per=shape['partition_tiles'];result={}
    assert per>0 and 0<=rank<world
    for s in shape['samples']:
        if s['mode']!='receiver_sparse':continue
        assert 0<=s['offset']<per and s['repeat']>=0
        tile=rank*per+s['offset'];raw=[r for r in s['receiver'] if r['tile']==tile]
        if len(raw)!=8*world:continue
        assert {(r['source'],r['fragment']) for r in raw}=={(r,f) for r in range(world) for f in range(8)}
        if any(not math.isfinite(r['observed_ns']) or r['observed_ns']<=0 for r in raw):continue
        poll=s['max_poll_cycle_ns']
        if not math.isfinite(poll) or poll<=0:continue
        if 'observer_status' in s and (s['observer_status'][2]!=0 or s['observer_status'][3]!=8*world):continue
        origin=0
        if relative:
            markers=s.get('local_markers',[])
            if len(markers)!=2 or not all(math.isfinite(x) for x in markers) or not 0<markers[0]<markers[1]:continue
            origin=markers[0]
            # A peer may publish before this rank starts. Keep signed offsets;
            # never compare absolute GPU clocks or absolute times across repeats.
        arrivals=[(max(r['observed_ns'] for r in raw if r['source']==src)-origin)/1000 for src in range(world)]
        key=(s['repeat']//per,tile)
        assert key not in result, ('duplicate pass/tile observation',key)
        result[key]=(arrivals,2*poll/1000)
    return result

def priority(train,world,score):
    """Only training observations enter the score. Zero means keep base order."""
    assert score in ('tail','gap') and len(train)==2
    gaps=[[a[r]-max(a[s] for s in range(world) if s!=r) for r in range(world)] for a,e in train]
    scores=[]
    for r in range(world):
        if score=='gap':
            admitted=all(g[r]>e for g,(a,e) in zip(gaps,train))
            value=statistics.median(g[r] for g in gaps)
        else:
            # Include near-tied critical sources, but require membership on BOTH
            # passes. Early tiles with a large skew must not outrank late joins.
            admitted=all(max(a)-a[r]<=e for a,e in train)
            value=statistics.median(max(a) for a,e in train)
        scores.append(max(0.,value) if admitted else 0.)
    return scores

def validate_mapping(rows,m,n,world,policy):
    tm,tn=m//128,(n+127)//128;tiles=tm*tn
    assert m%(128*world)==0 and len(rows)==world*tiles
    assert {r['rank'] for r in rows}==set(range(world))
    audits=[]
    for rank in range(world):
        rr=sorted([r for r in rows if r['rank']==rank],key=lambda r:r['index'])
        assert [r['index'] for r in rr]==list(range(tiles)), 'Padded/missing logical slots are unsupported'
        assert {(r['m'],r['n']) for r in rr}=={(x,y) for x in range(tm) for y in range(tn)}
        for r in rr:
            i=r['index'];dest=r['m']//(tm//world)
            assert r['destination']==dest
            if policy=='interleaved':
                within=i//world
                assert (r['m'],r['n'])==(((i%world+rank)%world)*(tm//world)+within//tn,within%tn)
            else:
                assert (r['m'],r['n'])==((r['base_m']+tm//world*(rank+1))%tm,r['base_n'])
        local=[r['index'] for r in rr if r['destination']==rank]
        remote=[r['index'] for r in rr if r['destination']!=rank]
        audits.append(dict(rank=rank,first_local_slot=min(local),last_remote_slot=max(remote),
                           all_remote_before_local=max(remote)<min(local)))
    return audits

def main(source,out,window=16,score="tail"):
    assert window>0 and score in ("tail","gap")
    cfg=json.loads((source/'calibration-complete.json').read_text());assert cfg['completed']
    world=cfg['config'].get('world_size',4);assert world in (4,8)
    result=dict(schema_version=2,score_mode=score,planner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),world_size=world,calibration=str(source.resolve()),window=window,training_passes=[0,1],validation_passes=[2],
       score=('median receiver-local join time; sources near-critical in both passes' if score=='tail' else 'median last-source gap; resolved last source must agree in both passes'),
       sampling_limitation='Exploratory sampled labels; no low-perturbation mechanism claim. Performance is tested independently.',policies={},inputs={})
    for policy in ['remote_first','interleaved']:
        reports=[json.loads((source/f'b0-{policy}-instrumented-rank{r}.json').read_text()) for r in range(world)]
        assert all(r['completed'] for r in reports)
        assert all(r.get('rank',i)==i and r.get('world_size',world)==world and r.get('policy',policy)==policy for i,r in enumerate(reports))
        plans=[]
        for si,(m,n,k) in enumerate(cfg['config']['shapes']):
            tn=(n+127)//128;tiles=m//128*tn
            with (source/f'mapping-{policy}-{m}-{n}-{k}.csv').open() as f:
                rows=list(csv.DictReader(f))
            rows=[{k:int(v) for k,v in r.items()} for r in rows]
            audit=validate_mapping(rows,m,n,world,policy)
            scores=[{} for _ in range(world)];valid=0;heldout=[];over=[];admitted=0;validation_available=0
            for target,report in enumerate(reports):
                sh=report['shapes'][si];assert (sh['M'],sh['N'],sh['K_global'])==(m,n,k)
                assert sh['partition_tiles']==tiles//world
                obs=observations(sh,target,world,relative=score=='tail')
                for tile in range(target*tiles//world,(target+1)*tiles//world):
                    if not all((p,tile) in obs for p in [0,1]):continue
                    valid+=1
                    train=[obs[(p,tile)] for p in [0,1]]
                    values=priority(train,world,score)
                    candidates={r for r in range(world) if values[r]>0}
                    admitted+=bool(candidates)
                    for r in range(world):scores[r][tile]=values[r]
                    if (2,tile) in obs:
                        validation_available+=1
                        a,e=obs[(2,tile)];winner=max(range(world),key=lambda r:a[r])
                        if candidates and a[winner]-max(a[r] for r in range(world) if r!=winner)>e:
                            heldout.append(winner in candidates)
                off={s['repeat']:s['forward_us'] for s in sh['samples'] if s['mode']=='off'}
                over += [100*(s['forward_us']/off[s['repeat']]-1) for s in sh['samples'] if s['mode']=='receiver_sparse']
            maps=[];coords=[];changed=[]
            for rank in range(world):
                rr=sorted([dict(r,tn=tn) for r in rows if r['rank']==rank],key=lambda r:r['index'])
                mapping=permute(rr,scores[rank],window)
                maps.append(mapping);coords.append([rr[j]['m']*tn+rr[j]['n'] for j in mapping])
                changed.append(sum(i!=j for i,j in enumerate(mapping)))
            # No reliable signal is a valid identity fallback, not a fitting error.
            plans.append(dict(M=m,N=n,K_global=k,K_local=k//world,tiles=tiles,maps=maps,coords=coords,
              changed_per_rank=changed,identity_fallback=not any(changed),base_mapping_audit=audit,
              training_valid_tiles=valid,training_admitted_tiles=admitted,total_tiles=tiles,
              validation_available_tiles=validation_available,
              heldout_metric="resolved heldout winner in training-admitted source set; not tail-rank accuracy",
              heldout_resolved_tiles=len(heldout),heldout_critical_set_hit_rate=statistics.mean(heldout) if heldout else None,
              heldout_last_source_accuracy=statistics.mean(heldout) if heldout else None,
              sampled_rank_median_overhead_percent=statistics.median(over) if over else None))
        result['policies'][{'remote_first':'remote_arrival','interleaved':'interleaved_arrival'}[policy]]=dict(base=policy,shapes=plans)
    for p in [*source.glob('b0-*-rank*.json'),*source.glob('mapping-*.csv'),source/'calibration-complete.json']:
        result['inputs'][str(p.resolve())]=hashlib.sha256(p.read_bytes()).hexdigest()
    out.write_text(json.dumps(result,indent=2)+'\n')
    for p,v in result['policies'].items():
        for s in v['shapes']:print(p,s['M'],s['N'],s['K_global'],s['changed_per_rank'],s['heldout_last_source_accuracy'])
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('source',type=Path);a.add_argument('out',type=Path);a.add_argument('--window',type=int,default=16);a.add_argument('--score',choices=['tail','gap'],default='tail');v=a.parse_args();main(v.source,v.out,v.window,v.score)
