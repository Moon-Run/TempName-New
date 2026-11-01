"""CPU scheduling/statistics fixtures; no generated fixture is a GPU result."""
from collections import Counter
import copy
import itertools
import unittest
from unittest.mock import patch

import op1_abc_protocol as p


class Protocol(unittest.TestCase):
    def test_complete_counterbalanced_cycles(self):
        for trials, blocks in ((1, 8), (2, 16)):
            rows = p.schedule(trials, blocks)
            self.assertEqual(len(rows), trials*blocks*4)
            for trial in range(1, trials+1):
                own = [row for row in rows if row['trial'] == trial]
                orders = [tuple(row['arm'] for row in own if row['block'] == block) for block in range(blocks)]
                for position in range(4):
                    self.assertEqual(Counter(order[position] for order in orders),
                                     Counter(dict.fromkeys(p.ARMS, blocks//4)))
                for a, b in itertools.combinations(p.ARMS, 2):
                    self.assertEqual(sum(order.index(a) < order.index(b) for order in orders), blocks//2)
                self.assertEqual(Counter(pair for order in orders for pair in zip(order, order[1:])),
                                 Counter({pair: blocks//4 for pair in itertools.permutations(p.ARMS, 2)}))
        self.assertEqual(p.schedule(), p.schedule())
        self.assertNotEqual(p.PROTOCOL, p.base.PROTOCOL)
        self.assertEqual([job['mode'] for job in p.jobs()[:8]], ['smoke']*4+['profile']*4)

    def test_reject_unsupported_or_invalid_cycles(self):
        for trials, blocks in ((0, 8), (3, 8), (1, 0), (1, 6), (1, 9), (True, 8)):
            with self.assertRaises(ValueError):
                p.schedule(trials, blocks)

    def test_statistics_retain_negative_gains_and_every_window(self):
        samples = [dict(job, ms_per_step=1030-10*p.ARMS.index(job['arm']),
            peak_allocated_gib=17, peak_reserved_gib=18, native_workspace_gib=1,
            peak_allocated_plus_taco_workspace_gib=18) for job in p.schedule(blocks=8)]
        with patch.object(p, 'definition', side_effect=lambda arm: {'arm': arm}):
            result = p.report(samples, blocks=8)
        self.assertEqual(result['timing_windows'], 32)
        self.assertEqual(result['formal_optimizer_steps'], 1600)
        self.assertEqual(result['timing_rank_records'], 256)
        self.assertTrue(result['counterbalanced_order'])
        self.assertTrue(result['pairwise_ci_available'])
        comparison = result['comparisons']['compare1']
        self.assertAlmostEqual(comparison['arms'][p.ARMS[0]]['reduction_vs_improved_percent'], -3)
        self.assertLess(comparison['pairs'][p.ARMS[0]+'/flux-improved']['percent'], 0)
        self.assertEqual(comparison['pairs'][p.ARMS[0]+'/flux-improved']['faster_blocks'], 0)
        self.assertEqual(len(comparison['pairs']), 6)
        self.assertIn('选择性量化', p.markdown(result))
        self.assertIn('no original cross-GPU tile-arrival overlap', result['limitation'])
        for bad in (samples[:-1], [samples[1], samples[0], *samples[2:]]):
            with self.assertRaises(ValueError):
                p.report(bad, blocks=8)
        bad = copy.deepcopy(samples)
        bad[0]['ms_per_step'] = float('nan')
        with self.assertRaises(ValueError):
            p.report(bad, blocks=8)

    def test_default_three_blocks_finish_without_bootstrap_or_extra_blocks(self):
        for trials in (1, 2):
            jobs = p.schedule(trials)
            self.assertEqual(len(jobs), trials*12)
            for trial in range(1, trials+1):
                own = [job for job in jobs if job['trial'] == trial]
                self.assertEqual({job['block'] for job in own}, {0, 1, 2})
                self.assertEqual(own, p.schedule(trials, 8)[(trial-1)*32:(trial-1)*32+12])
                for block in range(3):
                    self.assertEqual({job['arm'] for job in own if job['block'] == block}, set(p.ARMS))
            samples = [dict(job, ms_per_step=1030-10*p.ARMS.index(job['arm']),
                peak_allocated_gib=17, peak_reserved_gib=18, native_workspace_gib=1,
                peak_allocated_plus_taco_workspace_gib=18) for job in jobs]
            with patch.object(p, 'paired', side_effect=AssertionError('No short-run bootstrap')):
                result = p.report(samples, trials=trials)
            self.assertEqual(result['formal_optimizer_steps'], trials*600)
            self.assertEqual(result['timing_rank_records'], trials*96)
            self.assertTrue(result['completed'])
            self.assertFalse(result['counterbalanced_order'])
            self.assertFalse(result['pairwise_ci_available'])
            for comparison in result['comparisons'].values():
                self.assertEqual(len(comparison['pairs']), 6)
                for pair in comparison['pairs'].values():
                    self.assertEqual(pair['paired_blocks'], 3)
                    self.assertIsNone(pair['ci95'])
                self.assertAlmostEqual(comparison['pairs'][p.ARMS[0]+'/flux-improved']['percent'], -3)
            self.assertIn('三块短测', p.markdown(result))
            self.assertIn('不计算 95% CI', p.markdown(result))
            with self.assertRaises(ValueError):
                p.report(samples[:-1], trials=trials)

    def test_window_is_slowest_rank_not_mean_rank(self):
        rows = [dict(rank=rank, timed_steps=50, elapsed_seconds=40+rank,
                     peak_allocated_gib=17, peak_reserved_gib=18, native_workspace_bytes=2**30)
                for rank in range(8)]
        for arm in p.ARMS:
            job = next(job for job in p.schedule() if job['arm'] == arm)
            sample = p.summarize(rows, job)
            self.assertEqual(sample['arm'], arm)
            self.assertEqual(sample['ms_per_step'], 940)
            self.assertEqual(sample['peak_allocated_plus_taco_workspace_gib'], 18)


if __name__ == '__main__':
    unittest.main()
