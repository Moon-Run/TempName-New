"""Check regression sign convention and cancellation of common timing drift."""
import unittest
from report import change


class RegressionTests(unittest.TestCase):
    def test_positive_means_slower(self):
        result=change([10,20,30,40,50,60],[11,22,33,44,55,66])
        self.assertAlmostEqual(result['increase_percent'],10)
        for bound in result['ci95']:self.assertAlmostEqual(bound,10)

    def test_common_drift_cancels(self):
        candidate=[9,10,11,12,13,14];reference=[10,11,12,13,14,15]
        before=[a/b for a,b in zip(candidate,reference)]
        after=[(1.1*a)/(1.1*b) for a,b in zip(candidate,reference)]
        result=change(before,after)
        self.assertAlmostEqual(result['increase_percent'],0)
        for bound in result['ci95']:self.assertAlmostEqual(bound,0)


if __name__=='__main__':unittest.main()
