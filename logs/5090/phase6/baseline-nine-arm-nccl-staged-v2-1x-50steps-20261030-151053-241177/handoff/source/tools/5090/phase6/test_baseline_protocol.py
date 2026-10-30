"""CPU-only protocol regressions; fabricated samples are never GPU evidence."""
from collections import Counter
import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('rtx5090_protocol_test', Path(__file__).with_name('baseline_protocol.py'))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class ScheduleTests(unittest.TestCase):
    def test_position_precedence_and_adjacent_pair_balance(self):
        windows = p.schedule()
        self.assertEqual(len(windows), 162)
        self.assertEqual(len({j['directory'] for j in p.jobs()}), 180)
        positions, adjacent, precedence = Counter(), Counter(), Counter()
        for block in range(18):
            group = [j for j in windows if j['block'] == block]
            self.assertEqual({j['arm'] for j in group}, set(p.ARMS))
            order = [j['arm'] for j in group]
            for slot, arm in enumerate(order):
                positions[arm, slot] += 1
                for later in order[slot + 1:]:
                    precedence[arm, later] += 1
            adjacent.update(zip(order, order[1:]))
        self.assertEqual(set(positions.values()), {2})
        self.assertEqual(len(positions), 81)
        self.assertEqual(set(precedence.values()), {9})
        self.assertEqual(len(precedence), 72)
        self.assertEqual(set(adjacent.values()), {2})
        self.assertEqual(len(adjacent), 72)

    def test_invalid_and_repeated_trials(self):
        for trials, blocks in ((0, 18), (3, 18), (True, 18), (1, 0), (1, 8), (1, 19), (1, 18.0)):
            with self.assertRaises(ValueError):
                p.schedule(trials, blocks)
        windows = p.schedule(2, 36)
        self.assertEqual(len(windows), 648)
        self.assertEqual(len({j['directory'] for j in windows}), 648)
        self.assertEqual(windows, p.schedule(2, 36))


class StatisticsTests(unittest.TestCase):
    @staticmethod
    def samples():
        # MA11 is deliberately slower than every other arm and has one slow
        # window; neither negative outcomes nor outliers may be discarded.
        return [dict(j, ms_per_step=(900 if j['block'] == 0 else 650) if j['arm'] == 'ma11' else 500,
                     peak_allocated_gib=18.0, peak_reserved_gib=20.0, native_workspace_gib=2.0,
                     peak_allocated_plus_taco_workspace_gib=20.0) for j in p.schedule()]

    def test_rank_maximum_and_reject_incomplete_window(self):
        job = p.schedule()[0]
        rows = [dict(rank=i, elapsed_seconds=25 + i, timed_steps=50,
                     peak_allocated_gib=17 + i / 10, peak_reserved_gib=19 + i / 10,
                     native_workspace_bytes=2*2**30 if i == 0 else 0) for i in range(8)]
        summary = p.summarize(rows, job)
        self.assertEqual(summary['ms_per_step'], 640)
        self.assertEqual(summary['peak_allocated_gib'], 17.7)
        self.assertEqual(summary['native_workspace_gib'], 2)
        self.assertEqual(summary['peak_allocated_plus_taco_workspace_gib'], 19)
        with self.assertRaises(ValueError):
            p.summarize(rows[:-1], job)
        rows[-1]['timed_steps'] = 20
        with self.assertRaises(ValueError):
            p.summarize(rows, job)

    def test_all_windows_and_negative_effect_retained(self):
        samples = self.samples()
        result = p.report(samples)
        self.assertEqual(result['all_samples'], samples)
        self.assertEqual(result['timing_rank_records'], 1296)
        self.assertEqual(result['formal_optimizer_steps'], 8100)
        arms = result['comparisons']['compare1']['arms']
        self.assertAlmostEqual(arms['ma11']['mean_ms'], (17 * 650 + 900) / 18)
        self.assertEqual(arms['ma11']['max_ms'], 900)
        self.assertLess(arms['ma89']['ma11_mean_reduction_percent'], 0)
        pairs = result['comparisons']['compare1']['pairs']
        self.assertEqual(len(pairs), 36)
        self.assertLess(pairs['ma11/ma89']['ci95'][1], 0)
        self.assertEqual(pairs['ma11/ma89']['paired_blocks'], 18)
        self.assertIn('900.000', p.markdown(result))
        self.assertEqual(result['calibration_origin'], 'frozen_h100_arrival_tables_and_quantization_masks')
        self.assertTrue(result['original_mechanisms_not_identical'])
        self.assertFalse(result['original_peer_overlap_preserved'])
        self.assertEqual(result['transport'], p.TRANSPORT)
        self.assertNotEqual(result['protocol'], 'rtx5090-phase6-nine-arm-v1')

    def test_incomplete_reordered_and_nonfinite_reports_rejected(self):
        samples = self.samples()
        variants = [samples[:-1], samples[1:] + samples[:1], samples + samples[:1]]
        bad = copy.deepcopy(samples)
        bad[0]['ms_per_step'] = float('nan')
        variants.append(bad)
        bad = copy.deepcopy(samples)
        bad[0]['peak_allocated_gib'] = 0
        variants.append(bad)
        for records in variants:
            with self.assertRaises(ValueError):
                p.report(records)

    def test_paired_constant_ratio_and_invalid_values(self):
        result = p.paired([400] * 18, [500] * 18)
        self.assertAlmostEqual(result['percent'], 20.0)
        for bound in result['ci95']:
            self.assertAlmostEqual(bound, 20.0)
        with self.assertRaises(ValueError):
            p.paired([0] * 18, [500] * 18)
        with self.assertRaises(ValueError):
            p.paired([400] * 17, [500] * 18)


if __name__ == '__main__':
    unittest.main()
