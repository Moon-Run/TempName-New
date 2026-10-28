import unittest
from unittest.mock import patch
from plan import select, fit


class SelectionTests(unittest.TestCase):
    def test_budget_and_local_exclusion(self):
        scores=[[float(i+1) for i in range(256)] for _ in range(4)]
        mask=select(scores,4,.25)
        self.assertEqual([sum(r) for r in mask],[48]*4)
        for src,row in enumerate(mask): self.assertFalse(any(row[src*64:(src+1)*64]))

    def test_no_signal_preserves_bf16(self):
        self.assertEqual(select([[0.]*256 for _ in range(4)],4,.25),[[0]*256 for _ in range(4)])

    def test_ties_are_deterministic(self):
        mask=select([[1.]*256 for _ in range(4)],4,.25)
        self.assertEqual([i for i,v in enumerate(mask[0]) if v],list(range(64,112)))

    def test_heldout_does_not_choose_mask(self):
        reports=[dict(shapes=[dict(M=2048,N=2048,K_global=8192,partition_tiles=64,
                   samples=[dict(mode='off',repeat=0,forward_us=10),dict(mode='receiver_sparse',repeat=0,forward_us=12)])]) for _ in range(4)]
        def observations(sh,target,world,relative):
            return {(p,t):([10.+t,11.+t,20.+t,12.+t],1.) for p in range(3) for t in range(target*64,(target+1)*64)}
        with patch('plan.arrival.observations',side_effect=observations): a=fit(reports)
        def changed(sh,target,world,relative):
            result=observations(sh,target,world,relative)
            for t in range(target*64,(target+1)*64):result[(2,t)]=([1000.,0.,0.,0.],1.)
            return result
        with patch('plan.arrival.observations',side_effect=changed): b=fit(reports)
        self.assertEqual(a['mask'],b['mask'])
        self.assertNotEqual(a['heldout_near_critical_fraction'],b['heldout_near_critical_fraction'])


if __name__=='__main__':unittest.main()
