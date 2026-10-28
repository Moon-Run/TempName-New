"""Apply the existing phase6 tail/selection rules to eight-rank BF16 calibration."""
import argparse,csv,hashlib,importlib.util,json,statistics,sys
from pathlib import Path
REPO=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from model_config import DEFAULT_MODEL, read_model, mlp_shape
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
arrival=load('tp8_arrival',REPO/'tools/a800/arrival/plan.py')
selection=load('tp8_selection',REPO/'tools/a800/phase5/plan.py')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main(source,out,model_path=None):
 source,out=source.resolve(),out.resolve();cfg=json.loads((source/'calibration-complete.json').read_text());assert cfg['completed'] and cfg['config']['world_size']==8
 model=read_model(model_path,8) if model_path else None
 world=8;window=64;fraction=1/64;m,n,k=mlp_shape(model) if model else [2048,2048,8192];tiles=m//128*(n//128);per=tiles//world;nt=n//128
 assert [m,n,k] in cfg['config']['shapes'], 'Fresh BF16 calibration for the requested model shape is required'
 result=dict(schema=2,world=world,window=window,budget_fraction=fraction,training_passes=[0,1],heldout_passes=[2],shape_catalog=[[m,n,k]],policies={},inputs={},limitation='Existing same-allocation TP8 BF16 samples; phase6 rules with fixed window64 and budget1/64. No tuning, no post-quantization or post-reorder recalibration.')
 for policy in ['remote_first','interleaved']:
  paths=[source/f'b0-{policy}-instrumented-rank{r}.json' for r in range(world)]
  reports=[json.loads(p.read_text()) for p in paths]
  scores=[[0.]*tiles for _ in range(world)];heldout={};overhead=[];invalid=[];valid=0
  for rank,report in enumerate(reports):
   assert report['completed'] and report['rank']==rank
   sh=next(x for x in report['shapes'] if [x['M'],x['N'],x['K_global']]==[m,n,k]);assert sh['partition_tiles']==per
   obs=arrival.observations(sh,rank,world,relative=True)
   for tile in range(rank*per,(rank+1)*per):
    invalid.extend(dict(receiver=rank,pass_index=p,tile=tile) for p in (0,1,2) if (p,tile) not in obs)
    if all((p,tile) in obs for p in (0,1)):
     valid+=1;values=arrival.priority([obs[(p,tile)] for p in (0,1)],world,'tail')
     for src in range(world):scores[src][tile]=values[src]
    if (2,tile) in obs:heldout[tile]=obs[(2,tile)]
   off={x['repeat']:x['forward_us'] for x in sh['samples'] if x['mode']=='off'}
   overhead.extend(100*(x['forward_us']/off[x['repeat']]-1) for x in sh['samples'] if x['mode']=='receiver_sparse')
  mapping=source/f'mapping-{policy}-{m}-{n}-{k}.csv';paths.append(mapping)
  rows=[{key:int(value) for key,value in row.items()} for row in csv.DictReader(mapping.open())]
  arrival.validate_mapping(rows,m,n,world,policy)
  maps=[];coords=[];base_coords=[];changed=[]
  for rank in range(world):
   rr=sorted([dict(x,tn=nt) for x in rows if x['rank']==rank],key=lambda x:x['index']);base=[x['m']*nt+x['n'] for x in rr]
   order=arrival.permute(rr,dict(enumerate(scores[rank])),window)
   slots={d:[i for i,x in enumerate(rr) if x['destination']==d] for d in range(world)}
   position={i:(d,j//window) for d,indices in slots.items() for j,i in enumerate(indices)}
   assert all(position[i]==position[j] for i,j in enumerate(order))
   maps.append(order);coords.append([base[i] for i in order]);base_coords.append(base);changed.append(sum(i!=j for i,j in enumerate(order)))
  mask=selection.select(scores,world,fraction)
  assert all(not any(row[src*per:(src+1)*per]) for src,row in enumerate(mask))
  assert all(sum(row)<=int(tiles*(world-1)/world*fraction) for row in mask)
  chosen=[(src,tile) for src in range(world) for tile in range(tiles) if mask[src][tile]]
  hits=[max(heldout[t][0])-heldout[t][0][src]<=heldout[t][1] for src,t in chosen if t in heldout]
  entry=dict(shape=[m,n,k],M=m,N=n,K_global=k,K_local=k//world,tiles=tiles,maps=maps,coords=coords,base_coords=base_coords,changed_per_rank=changed,mask=mask,selected_per_source=list(map(sum,mask)),training_valid_tiles=valid,invalid_observations=invalid,heldout_selected_observations=len(hits),selected_remote_fraction=len(chosen)/((world-1)*tiles),heldout_near_critical_fraction=statistics.mean(hits) if hits else None,sampled_median_overhead_percent=statistics.median(overhead))
  result['policies'][policy]=[entry]
  for p in paths:result['inputs'][str(p)]=sha(p)
  print(policy,'valid',valid,'changed',changed,'selected',entry['selected_per_source'],'fraction',entry['selected_remote_fraction'],flush=True)
 if model_path:result['inputs'][str(model_path.resolve())]=sha(model_path)
 for p in [source/'calibration-complete.json',Path(__file__),REPO/'tools/a800/arrival/plan.py',REPO/'tools/a800/phase5/plan.py']:result['inputs'][str(p.resolve())]=sha(p)
 out.parent.mkdir(parents=True,exist_ok=True);assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('out',type=Path)
 p.add_argument('--model-config',type=Path,default=DEFAULT_MODEL)
 a=p.parse_args();main(a.source,a.out,a.model_config)
