import unittest
from summarize import join_observation
class ObservationTests(unittest.TestCase):
 def records(self,times):return [dict(source=s,fragment=f,observed_ns=t-f) for s,t in enumerate(times) for f in range(8)]
 def test_four_complete_sources(self):
  r=join_observation(self.records([100,200,300,400]),4,8,10,50,500)
  self.assertEqual(r['last_source'],3);self.assertTrue(r['last_source_resolved'])
 def test_near_tie_is_unresolved_despite_large_skew(self):
  self.assertFalse(join_observation(self.records([100,200,395,400]),4,8,10,50,500)['last_source_resolved'])
 def test_first_scan_is_censored(self):
  self.assertIsNone(join_observation(self.records([-100,200,300,400]),4,8,10,50,500))
 def test_missing_fragment_rejected(self):
  with self.assertRaises(AssertionError):join_observation(self.records([100,200,300,400])[:-1],4,8,10,50,500)
 def test_duplicate_fragment_rejected(self):
  rs=self.records([100,200,300,400])
  with self.assertRaises(AssertionError):join_observation(rs+[rs[0]],4,8,10,50,500)
if __name__=='__main__':unittest.main()
