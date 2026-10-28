import tempfile,unittest
from pathlib import Path
from summarize_operator import result_record_names, trial_maxima
class ResultFilesTest(unittest.TestCase):
 def test_profiles_excluded_and_extra_rank_preserved(self):
  with tempfile.TemporaryDirectory() as tmp:
   p=Path(tmp)
   for name in ['window-00-remote_arrival-rank0.json','window-00-remote_arrival-rank0-profile.json','window-00-remote_arrival-rank1.json','window-00-remote_arrival-rank4.json']:
    (p/name).write_text('{}')
   self.assertEqual(result_record_names(p),{'window-00-remote_arrival-rank0.json','window-00-remote_arrival-rank1.json','window-00-remote_arrival-rank4.json'})
class TimingTest(unittest.TestCase):
 def fixture(self):
  cfg=dict(iters=100,trial_count=2,warmup_initial=500,warmup_per_trial=30)
  return cfg,[dict(cfg,trials_us=[10+r,20-r]) for r in range(8)]
 def test_max_per_trial_includes_eighth_rank(self):
  cfg,rows=self.fixture();self.assertEqual(trial_maxima(rows,cfg),[17,20])
 def test_nan_negative_and_inf_on_any_rank_rejected(self):
  for value in (float('nan'),float('inf'),-1.,0.):
   cfg,rows=self.fixture();rows[-1]['trials_us'][0]=value
   with self.assertRaises(AssertionError):trial_maxima(rows,cfg)
 def test_mismatched_iteration_count_rejected(self):
  cfg,rows=self.fixture();rows[-1]['iters']=1
  with self.assertRaises(AssertionError):trial_maxima(rows,cfg)
if __name__=='__main__':unittest.main()
