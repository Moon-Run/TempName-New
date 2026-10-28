"""Reject DP2 or incomplete groups masquerading as a TP8 result."""
import unittest
from protocol import POLICIES,orders,validate_topology


def rows():
    return [dict(rank=i,local_rank=i%4,tp_rank=i,dp_rank=0,host=f'node{i//4}',
        gpu_uuid=f'gpu{i}',tokens_sha256='same',global_tokens_sha256='global',
        policy='original',tp_group_ranks=list(range(8)),dp_group_ranks=[i],
        local_group_ranks=list(range(i//4*4,i//4*4+4))) for i in range(8)]


class ProtocolTests(unittest.TestCase):
    def test_valid(self):validate_topology(rows())
    def test_reject_tp4_dp2(self):
        x=rows()
        for r in x:r.update(tp_rank=r['rank']%4,dp_rank=r['rank']//4)
        with self.assertRaises(AssertionError):validate_topology(x)
    def test_reject_bad_group_and_data(self):
        for key,value in [('tp_group_ranks',[0,1,2,3]),('tokens_sha256','different'),('gpu_uuid','gpu0')]:
            x=rows();x[4][key]=value
            with self.assertRaises(AssertionError):validate_topology(x)
    def test_orders(self):
        a=orders(1)
        for i in range(4):self.assertEqual({r[i] for r in a},set(POLICIES))
        self.assertEqual(orders(2),[list(reversed(r)) for r in reversed(a)])


if __name__=='__main__':unittest.main()
