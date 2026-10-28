import csv,json,tempfile,unittest
from pathlib import Path
from plan import main,observations,permute
class EightRankTests(unittest.TestCase):
 def test_eight_sources_and_censoring(self):
  raw=[dict(source=s,tile=0,fragment=f,observed_ns=10000+s*1000+f) for s in range(8) for f in range(8)]
  sample=dict(mode='receiver_sparse',repeat=0,offset=0,max_poll_cycle_ns=10,receiver=raw)
  shape=dict(partition_tiles=8,samples=[sample])
  a,e=observations(shape,0,8)[(0,0)]
  self.assertEqual(len(a),8);self.assertEqual(a[7],17.007)
  raw.pop();self.assertEqual(observations(shape,0,8),{})
 def test_fit_all_eight_ranks_and_holdout(self):
  with tempfile.TemporaryDirectory() as tmp:
   out=Path(tmp);tp=8;per=8;tiles=tp*per
   (out/'calibration-complete.json').write_text(json.dumps(dict(completed=True,config=dict(world_size=tp,shapes=[[1024,1024,2048]]))))
   for policy in ['remote_first','interleaved']:
    with (out/f'mapping-{policy}-1024-1024-2048.csv').open('w') as f:
     w=csv.DictWriter(f,fieldnames=['rank','index','m','n','destination','base_m','base_n']);w.writeheader()
     for rank in range(tp):
      for i in range(tiles):
       dest=(i%tp+rank)%tp if policy=='interleaved' else (i//per+rank+1)%tp
       col=i//tp if policy=='interleaved' else i%per
       w.writerow(dict(rank=rank,index=i,m=dest,n=col,destination=dest,base_m=i//per,base_n=i%per))
    for rank in range(tp):
     samples=[]
     for repeat in range(3*per):
      offset=repeat%per;tile=rank*per+offset;winner=tile%tp
      raw=[dict(source=s,tile=tile,fragment=f,observed_ns=10000+f+(1000*(offset+1) if s==winner else s*10)) for s in range(tp) for f in range(8)]
      samples.extend([dict(mode='off',repeat=repeat,forward_us=100),dict(mode='receiver_sparse',repeat=repeat,forward_us=103,offset=offset,max_poll_cycle_ns=10,local_markers=[1000,200000],receiver=raw)])
     (out/f'b0-{policy}-instrumented-rank{rank}.json').write_text(json.dumps(dict(completed=True,shapes=[dict(M=1024,N=1024,K_global=2048,partition_tiles=per,samples=samples)])))
   main(out,out/'plan.json',window=4);plan=json.loads((out/'plan.json').read_text())
   self.assertEqual(plan['world_size'],8)
   for v in plan['policies'].values():
    shape=v['shapes'][0];self.assertEqual(shape['K_local'],256)
    self.assertEqual(len(shape['maps']),8)
    self.assertTrue(any(shape['changed_per_rank']))
    for mapping in shape['maps']:self.assertEqual(sorted(mapping),list(range(tiles)))
    self.assertEqual(shape['heldout_last_source_accuracy'],1)
if __name__=='__main__':unittest.main()
