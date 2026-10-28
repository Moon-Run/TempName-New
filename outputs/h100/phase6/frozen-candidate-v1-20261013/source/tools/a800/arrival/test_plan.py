import unittest
import csv,json,tempfile
from pathlib import Path
from plan import permute, observations, main, priority, validate_mapping
class PlanTest(unittest.TestCase):
 def test_destination_budget_and_stable_ties(self):
  rows=[dict(index=i,m=i%4,n=i//4,tn=8,destination=i%4) for i in range(32)]
  scores={r['m']*8+r['n']:float(r['n']) for r in rows}
  p=permute(rows,scores,4)
  self.assertEqual(sorted(p),list(range(32)))
  for i,j in enumerate(p):
   self.assertEqual(i%4,j%4)
   self.assertEqual((i//4)//4,(j//4)//4)
  self.assertEqual(p[:4],[12,13,14,15])
  self.assertEqual(permute(rows,{},4),list(range(32)))
 def test_short_tail_and_missing_scores(self):
  rows=[dict(index=i,m=0,n=i,tn=9,destination=0) for i in range(9)]
  self.assertEqual(permute(rows,{2:9,8:20},4),[2,0,1,3,4,5,6,7,8])
 def test_reject_noncontiguous_input(self):
  with self.assertRaises(AssertionError):permute([dict(index=2,m=0,n=0,tn=1,destination=0)],{},4)
class ObservationTest(unittest.TestCase):
 def sample(self, bad=False):
  return dict(mode='receiver_sparse',repeat=0,offset=0,max_poll_cycle_ns=100,
   receiver=[dict(source=r,tile=0,fragment=f,observed_ns=(-1 if bad and r==0 and f==0 else 1000+r*100+f)) for r in range(4) for f in range(8)])
 def test_complete_contribution_and_uncertainty(self):
  got=observations(dict(partition_tiles=8,samples=[self.sample()]),0)
  self.assertEqual(got[(0,0)][0],[1.007,1.107,1.207,1.307])
  self.assertEqual(got[(0,0)][1],.2)
 def test_censored_observation_excluded(self):
  self.assertEqual(observations(dict(partition_tiles=8,samples=[self.sample(True)]),0),{})
 def test_duplicate_fragment_rejected(self):
  s=self.sample();s['receiver'][0]['fragment']=1
  with self.assertRaises(AssertionError):observations(dict(partition_tiles=8,samples=[s]),0)
class FittingTest(unittest.TestCase):
 def test_holdout_does_not_change_fitted_mapping(self):
  with tempfile.TemporaryDirectory() as tmp:
   source=Path(tmp);shape=[512,1024,2048];per=8;tiles=32
   (source/'calibration-complete.json').write_text(json.dumps(dict(completed=True,config=dict(shapes=[shape]))))
   for policy in ['remote_first','interleaved']:
    with (source/f'mapping-{policy}-512-1024-2048.csv').open('w') as f:
     w=csv.DictWriter(f,fieldnames=['rank','index','m','n','destination','base_m','base_n']);w.writeheader()
     for rank in range(4):
      for i in range(tiles):
       dest=(i%4+rank)%4 if policy=='interleaved' else (i//per+rank+1)%4
       col=i//4 if policy=='interleaved' else i%per
       w.writerow(dict(rank=rank,index=i,m=dest,n=col,destination=dest,base_m=i//per,base_n=i%per))
    for rank in range(4):
     samples=[]
     for repeat in range(3*per):
      offset=repeat%per;tile=rank*per+offset;winner=tile%4
      samples.append(dict(mode='off',repeat=repeat,forward_us=100))
      records=[dict(source=r,tile=tile,fragment=f,observed_ns=10000+f+(1000*(offset+1) if r==winner else r*20)) for r in range(4) for f in range(8)]
      samples.append(dict(mode='receiver_sparse',repeat=repeat,forward_us=103,offset=offset,max_poll_cycle_ns=10,local_markers=[1000,200000],receiver=records))
     row=dict(M=512,N=1024,K_global=2048,partition_tiles=per,samples=samples)
     (source/f'b0-{policy}-instrumented-rank{rank}.json').write_text(json.dumps(dict(completed=True,shapes=[row])))
   main(source,source/'plan1.json',window=8)
   first=json.loads((source/'plan1.json').read_text())
   for p in source.glob('b0-*-rank*.json'):
    r=json.loads(p.read_text())
    for sample in r['shapes'][0]['samples']:
     if sample['repeat']//per==2 and sample['mode']=='receiver_sparse':
      for rec in sample['receiver']:rec['observed_ns']=100000-rec['observed_ns']
    p.write_text(json.dumps(r))
   main(source,source/'plan2.json',window=8)
   second=json.loads((source/'plan2.json').read_text())
   for policy in first['policies']:
    a=first['policies'][policy]['shapes'][0];b=second['policies'][policy]['shapes'][0]
    self.assertEqual(a['maps'],b['maps'])
    self.assertNotEqual(a['heldout_last_source_accuracy'],b['heldout_last_source_accuracy'])
    self.assertTrue(any(a['changed_per_rank']))
   # Identical labels are a supported no-op; the old fitter raised here.
   for p in source.glob('b0-*-rank*.json'):
    r=json.loads(p.read_text())
    for sample in r['shapes'][0]['samples']:
     if sample['mode']=='receiver_sparse':
      for rec in sample['receiver']:rec['observed_ns']=50000+rec['fragment']
    p.write_text(json.dumps(r))
   main(source,source/'identity.json',window=8)
   for v in json.loads((source/'identity.json').read_text())['policies'].values():
    self.assertTrue(v['shapes'][0]['identity_fallback'])
    self.assertEqual(v['shapes'][0]['changed_per_rank'],[0]*4)
class PriorityRegressionTest(unittest.TestCase):
 def test_gap_rejects_inconsistent_last_source(self):
  # The old median admitted source 0: median(10, -1) == 4.5 > .2.
  train=[([20.,10.,9.,8.],.2),([10.,11.,9.,8.],.2)]
  self.assertEqual(priority(train,4,'gap'),[0.,0.,0.,0.])
  self.assertEqual(priority(train,4,'tail'),[0.,0.,0.,0.])
 def test_late_join_beats_large_early_skew(self):
  early=priority([([20.,10.,9.,8.],.2)]*2,4,'tail')
  late=priority([([100.,99.,98.,97.],.2)]*2,4,'tail')
  self.assertGreater(late[0],early[0])
 def test_near_tied_sources_remain_critical(self):
  a=priority([([100.,99.9,90.,80.],.2),([100.,100.1,90.,80.],.2)],4,'tail')
  self.assertGreater(a[0],0);self.assertEqual(a[0],a[1]);self.assertEqual(a[2:],[0.,0.])
 def test_relative_time_does_not_depend_on_absolute_clock(self):
  s=ObservationTest().sample();s['local_markers']=[500,5000]
  sh=dict(partition_tiles=8,samples=[s])
  original=observations(sh,0,relative=True)
  for rec in s['receiver']:rec['observed_ns']+=10**15
  s['local_markers']=[v+10**15 for v in s['local_markers']]
  self.assertEqual(observations(sh,0,relative=True),original)
 def test_missing_origin_and_zero_poll_are_not_training_labels(self):
  s=ObservationTest().sample();sh=dict(partition_tiles=8,samples=[s])
  self.assertEqual(observations(sh,0,relative=True),{})
  s['max_poll_cycle_ns']=0
  self.assertEqual(observations(sh,0),{})
 def test_duplicate_pass_tile_rejected(self):
  s=ObservationTest().sample()
  with self.assertRaises(AssertionError):observations(dict(partition_tiles=8,samples=[s,s]),0)
 def test_remote_offset_is_not_strict_remote_first(self):
  rows=[];tm=4;tn=8;world=4
  # Column-major base is used on a wide grid. Rank+1 wraps to local
  # before all columns of the remote partitions have been visited.
  for rank in range(world):
   for i in range(tm*tn):
    bm,bn=i%tm,i//tm;m=(bm+rank+1)%tm
    rows.append(dict(rank=rank,index=i,m=m,n=bn,destination=m,base_m=bm,base_n=bn))
  audit=validate_mapping(rows,512,1024,world,'remote_first')
  self.assertTrue(all(not r['all_remote_before_local'] for r in audit))
 def test_corrupted_destination_rejected(self):
  rows=[dict(rank=r,index=i,m=(i+r+1)%4,n=0,destination=(i+r+1)%4,base_m=i,base_n=0) for r in range(4) for i in range(4)]
  rows[0]['destination']=0
  with self.assertRaises(AssertionError):validate_mapping(rows,512,128,4,'remote_first')
if __name__=='__main__':unittest.main()
