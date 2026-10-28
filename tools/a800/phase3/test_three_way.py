"""Reject mapping defects that can otherwise make a three-policy comparison misleading."""
import copy
import itertools
import json
import unittest
from pathlib import Path

from summarize_extension import check_mapping_rows

class ThreeWayChecks(unittest.TestCase):
    def setUp(self):
        # Four destination partitions, each containing one row of two tiles.
        coordinates = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)]
        self.rows = [dict(index=i, m=m, n=n, destination=m) for i,(m,n) in enumerate(coordinates)]

    def check(self, rows, rank=0):
        return check_mapping_rows(rows, 512, 256, 4, rank, 'interleaved')

    def test_four_destinations(self):
        self.assertEqual(self.check(self.rows), [0,1,2,3,0,1,2,3])

    def test_rejects_duplicate_tile(self):
        rows = copy.deepcopy(self.rows)
        rows[4].update(m=0, n=0)
        with self.assertRaises(AssertionError):
            self.check(rows)

    def test_rejects_bijective_wrong_order(self):
        rows = copy.deepcopy(self.rows)
        rows[1]['index'], rows[2]['index'] = rows[2]['index'], rows[1]['index']
        with self.assertRaises(AssertionError):
            self.check(rows)

    def test_rejects_missing_destination(self):
        with self.assertRaises(AssertionError):
            self.check(self.rows[:-1])

    def test_rejects_wrong_rank_rotation(self):
        with self.assertRaises(AssertionError):
            self.check(self.rows, rank=1)

    def test_rejects_padded_logical_indices(self):
        rows = copy.deepcopy(self.rows)
        rows[-1]['index'] = 8
        with self.assertRaises(AssertionError):
            self.check(rows)

    def test_all_six_balanced_orders(self):
        config = json.loads(Path(__file__).with_name('three_way.json').read_text())
        self.assertEqual(set(map(tuple,config['orders'])), set(itertools.permutations(config['policies'])))
        for policy in config['policies']:
            for position in range(3):
                self.assertEqual(sum(order[position]==policy for order in config['orders']), 2)

if __name__ == '__main__':
    unittest.main()
