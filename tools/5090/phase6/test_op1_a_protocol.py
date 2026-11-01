"""CPU contracts: balanced three-arm scheduling and honest full-step statistics."""
from collections import Counter
import copy
import itertools
import unittest

import op1_a_protocol as p


class Protocol(unittest.TestCase):
    def test_complete_counterbalanced_cycles(self):
        for trials, blocks in ((1, 6), (2, 12)):
            rows = p.schedule(trials, blocks)
            self.assertEqual(len(rows), trials*blocks*3)
            for trial in range(1, trials+1):
                own = [row for row in rows if row['trial'] == trial]
                orders = [tuple(row['arm'] for row in own if row['block'] == block) for block in range(blocks)]
                self.assertEqual(Counter(orders), Counter({order: blocks//6 for order in itertools.permutations(p.ARMS)}))
                for index in range(3):
                    self.assertEqual(Counter(order[index] for order in orders), Counter(dict.fromkeys(p.ARMS, blocks//3)))
                for a, b in itertools.combinations(p.ARMS, 2):
                    self.assertEqual(sum(order.index(a) < order.index(b) for order in orders), blocks//2)
        self.assertEqual(p.schedule(), p.schedule())
        self.assertNotEqual(p.PROTOCOL, p.base.PROTOCOL)
        self.assertEqual(p.jobs()[:3], [dict(trial=0, block=-1, index=i, arm=a, mode='smoke', directory=f'smoke-{a}')
                                      for i, a in enumerate(p.ARMS)])

    def test_reject_partial_or_invalid_cycles(self):
        for trials, blocks in ((0, 6), (3, 6), (1, 0), (1, 5), (1, 7), (True, 6)):
            with self.assertRaises(ValueError):
                p.schedule(trials, blocks)

    def test_statistics_retain_negative_gains_and_every_window(self):
        # Synthetic CPU fixture, never written as experimental data.
        samples = [dict(job, ms_per_step={'a': 1020, 'flux-basic': 1000, 'flux-basic-interleaved': 990}[job['arm']],
            peak_allocated_gib=17, peak_reserved_gib=18, native_workspace_gib=1,
            peak_allocated_plus_taco_workspace_gib=18) for job in p.schedule()]
        result = p.report(samples)
        self.assertEqual(result['timing_windows'], 18)
        self.assertEqual(result['formal_optimizer_steps'], 900)
        self.assertEqual(result['timing_rank_records'], 144)
        comparison = result['comparisons']['compare1']
        self.assertAlmostEqual(comparison['arms']['flux-basic']['a_mean_reduction_percent'], -2)
        self.assertLess(comparison['pairs']['a/flux-basic']['percent'], 0)
        self.assertEqual(comparison['pairs']['a/flux-basic']['faster_blocks'], 0)
        self.assertIn('交替排序', p.markdown(result))
        for bad in (samples[:-1], [samples[1], samples[0], *samples[2:]]):
            with self.assertRaises(ValueError):
                p.report(bad)
        bad = copy.deepcopy(samples)
        bad[0]['ms_per_step'] = float('nan')
        with self.assertRaises(ValueError):
            p.report(bad)

    def test_window_is_slowest_rank_not_mean_rank(self):
        rows = [dict(rank=rank, timed_steps=50, elapsed_seconds=40+rank,
                     peak_allocated_gib=17, peak_reserved_gib=18, native_workspace_bytes=2**30)
                for rank in range(8)]
        job = next(job for job in p.schedule() if job['arm'] == 'flux-basic-interleaved')
        sample = p.summarize(rows, job)
        self.assertEqual(sample['arm'], 'flux-basic-interleaved')
        self.assertEqual(sample['ms_per_step'], 940)
        self.assertEqual(sample['peak_allocated_plus_taco_workspace_gib'], 18)


if __name__ == '__main__':
    unittest.main()
