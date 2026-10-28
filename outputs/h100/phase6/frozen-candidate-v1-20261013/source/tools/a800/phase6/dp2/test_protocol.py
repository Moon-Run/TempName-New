"""Reject wrong topology and silent missing DP synchronization before timing."""
import copy
import unittest

from protocol import POLICIES, orders, replica_checks, topology


def fixture():
    return [dict(rank=r, gpu_uuid=f'gpu-{r}', host=f'node-{r//4}',
                 local_rank=r%4, tp_rank=r%4, dp_rank=r//4,
                 tp_group_ranks=list(range(r//4*4,r//4*4+4)),
                 dp_group_ranks=[r%4,r%4+4], global_tokens_sha256='global',
                 tokens_sha256=f'data-{r//4}', initial_parameters_sha256=f'init-{r%4}',
                 final_parameters_sha256=f'final-{r%4}',
                 synced_gradients_sha256=f'grad-{r%4}',mode='smoke') for r in range(8)]


class ProtocolTests(unittest.TestCase):
    def test_topology(self):
        self.assertEqual(topology(8,4,4,2)['dp_groups'],[[0,4],[1,5],[2,6],[3,7]])
        for values in ((8,4,8,1),(4,4,4,1),(8,8,4,2)):
            with self.assertRaises(ValueError):topology(*values)

    def test_order_balanced_and_reversed(self):
        rows=orders(1)
        for index in range(6):
            self.assertEqual(set(row[index] for row in rows),set(POLICIES))
        self.assertEqual(orders(2),[list(reversed(row)) for row in reversed(rows)])

    def test_valid(self):
        replica_checks(fixture())

    def test_reject_divergence_and_wrong_groups(self):
        for key,value in [('final_parameters_sha256','diverged'),
                          ('synced_gradients_sha256','unsynced'),
                          ('initial_parameters_sha256','different'),
                          ('tp_group_ranks',[0,1,2,3]),('dp_group_ranks',[4,5]),
                          ('gpu_uuid','gpu-0')]:
            with self.subTest(key=key):
                rows=fixture();rows[4][key]=value
                with self.assertRaises(AssertionError):replica_checks(rows)

    def test_reject_duplicate_data(self):
        rows=fixture()
        for r in rows:r['tokens_sha256']='same-data'
        with self.assertRaises(AssertionError):replica_checks(rows)

    def test_missing_rank(self):
        with self.assertRaises(AssertionError):replica_checks(fixture()[:-1])


if __name__ == '__main__':unittest.main()
