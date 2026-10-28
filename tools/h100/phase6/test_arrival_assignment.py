"""Joint-tail regressions, exact optimizer checks, and evidence boundaries."""
import copy
import itertools
import json
import math
import random
import unittest

import arrival_tail
from arrival_assignment import (MAX_DISPLACEMENT, _bottleneck_assignment, _cost_matrix,
                                _fit_window, _safe_sample, fit_order)


def fixture(values, world=2, uncertainty=0.01, window=None, second=None):
    count = len(values)
    rows = [dict(rank=source, index=slot, m=(slot % world + source) % world,
                 n=slot // world, destination=(slot % world + source) % world)
            for source in range(world) for slot in range(count * world)]
    observations = {}
    for p in (0, 1, 2):
        pass_values = second if p == 1 and second is not None else values
        for destination in range(world):
            for tile, value in enumerate(pass_values):
                observations[p, destination * count + tile] = (
                    [float(value) + source * 0.5 + p * 0.1 for source in range(world)], uncertainty)
    return rows, observations, world, count if window is None else window


class ArrivalAssignmentTests(unittest.TestCase):
    def test_joint_tied_tails_escape_single_swap_stall(self):
        args = fixture([0, 0, 10, 10, 8, 8])
        self.assertTrue(arrival_tail.fit_order(*args)[1]['identity_fallback'])
        untouched = copy.deepcopy(args)
        maps, result = fit_order(*args)
        self.assertEqual(args, untouched)
        self.assertFalse(result['identity_fallback'])
        self.assertEqual(result['accepted_windows'], 4)
        for group in result['groups']:
            self.assertEqual(group['status'], 'candidate')
            candidate = group['candidate']
            self.assertEqual(candidate['changed_slots'], 4)
            self.assertTrue(all(abs(gain - 1) < 1e-12 for gain in candidate['gain_us']))
            self.assertTrue(all(gain > margin for gain, margin in
                                zip(candidate['gain_us'], candidate['required_margin_us'])))
            self.assertLess(candidate['bottleneck_cost_us'], 0)
        self.assertEqual(sorted(maps[0]), list(range(12)))
        json.dumps(result, allow_nan=False)

    def test_different_pass_tails_can_share_joint_candidate(self):
        _, result = fit_order(*fixture([0, 0, 10, 10, 8, 8], second=[0, 0, 8, 10, 10, 8]))
        self.assertFalse(result['identity_fallback'])
        for group in result['groups']:
            self.assertTrue(all(gain > margin for gain, margin in
                                zip(group['candidate']['gain_us'], group['candidate']['required_margin_us'])))

    def test_individual_pass_opportunities_can_conflict(self):
        maps, result = fit_order(*fixture([0, 0, 10, 10, 8, 8], second=[0, 0, 10, 8, 10, 8]))
        self.assertEqual(maps, [list(range(12))] * 2)
        for group in result['groups']:
            self.assertEqual(group['status'], 'inconsistent_training_identity')
            self.assertTrue(all(bound['clears_margin'] for bound in group['optimistic_bound']))
            self.assertFalse(group['unrestricted_joint_gate_feasible'])

    def test_monotone_and_flat_data_are_identity(self):
        for values in ([0, 1, 2, 3, 4, 5], [1] * 6, [1, 1, 1, 10, 10, 25], [20, 1]):
            with self.subTest(values=values):
                maps, result = fit_order(*fixture(values))
                self.assertTrue(result['identity_fallback'])
                self.assertTrue(all(g['status'] == 'no_model_gain_identity' for g in result['groups']))
                self.assertEqual(maps, [list(range(len(values) * 2))] * 2)

    def test_optimistic_bound_rejects_unresolved_gain_without_lowering_margin(self):
        for uncertainty in (1 / 6, 0.2, 2):
            _, result = fit_order(*fixture([0, 0, 10, 10, 8, 8], uncertainty=uncertainty))
            self.assertTrue(result['identity_fallback'])
            for group in result['groups']:
                self.assertEqual(group['status'], 'uncertainty_limited_identity')
                self.assertIsNone(group['candidate'])
                self.assertTrue(all(not bound['clears_margin'] for bound in group['optimistic_bound']))
        _, result = fit_order(*fixture([0, 0, 10, 10, 8, 8], uncertainty=0))
        self.assertFalse(result['identity_fallback'])

    def test_displacement_is_within_destination_positions(self):
        args = fixture([0, 0, 10, 10, 8, 8])
        for distance in (0, 1):
            _, result = fit_order(*args, max_displacement=distance)
            self.assertTrue(result['identity_fallback'])
            self.assertTrue(all(g['status'] == 'displacement_limited_identity' for g in result['groups']))
        _, result = fit_order(*args, max_displacement=2)
        self.assertFalse(result['identity_fallback'])
        self.assertTrue(all(g['max_displacement'] == 2 for g in result['groups']))

    def test_bottleneck_and_movement_ties_match_exhaustive_optimum(self):
        rng = random.Random(1849)
        for size in range(1, 7):
            for case in range(15):
                costs = [[None if rng.random() < 0.2 else rng.randrange(-3, 4)
                          for _ in range(size)] for _ in range(size)]
                possibilities = []
                for order in itertools.permutations(range(size)):
                    if any(costs[slot][tile] is None for slot, tile in enumerate(order)):
                        continue
                    key = (max(costs[slot][tile] for slot, tile in enumerate(order)),
                           sum(slot != tile for slot, tile in enumerate(order)),
                           sum(abs(slot - tile) for slot, tile in enumerate(order)))
                    possibilities.append(key)
                with self.subTest(size=size, case=case):
                    actual = _bottleneck_assignment(costs)
                    if not possibilities:
                        self.assertIsNone(actual)
                        continue
                    order, bottleneck = actual
                    key = (bottleneck, sum(slot != tile for slot, tile in enumerate(order)),
                           sum(abs(slot - tile) for slot, tile in enumerate(order)))
                    self.assertEqual(key, min(possibilities))
                    self.assertEqual(actual, _bottleneck_assignment(costs))
        self.assertEqual(_bottleneck_assignment([[0] * 6 for _ in range(6)]), (list(range(6)), 0))

    def test_surrogate_objective_and_optimistic_bound_match_enumeration(self):
        rng = random.Random(817)
        for _ in range(20):
            values, second = [[rng.randrange(20) for _ in range(5)] for _ in range(2)]
            rows, observations, world, _ = fixture(values, second=second, uncertainty=0)
            physical = list(range(5))
            samples = [_safe_sample(observations, p, physical, 0, world) for p in (0, 1)]
            order, result = _fit_window(samples, 2)
            all_orders = list(itertools.permutations(range(5)))
            maxima = [max(sample['arrivals']) for sample in samples]
            costs = _cost_matrix(samples, maxima, [0, 0], 2)
            optimum = min(max(costs[slot][tile] for slot, tile in enumerate(permutation))
                          for permutation in all_orders
                          if all(abs(slot - tile) <= 2 for slot, tile in enumerate(permutation)))
            self.assertEqual(result['status'] == 'candidate', optimum < -1e-12)
            if result['status'] == 'candidate':
                self.assertAlmostEqual(result['candidate']['bottleneck_cost_us'], optimum)
            for sample, bound in zip(samples, result['optimistic_bound']):
                best = min(max(arrival_tail._predicted(sample, permutation)) for permutation in all_orders)
                self.assertAlmostEqual(bound['after_max_us'], best)

    def test_missing_invalid_and_overflowing_samples_keep_identity_and_json_finite(self):
        rows, original, world, window = fixture([0, 0, 10, 10, 8, 8])
        for invalid in (None, ([math.nan, 0], .01), ([0, 0], -1), ([0, 0], math.inf),
                        ([0, 0], 1e308), ([0], .01), ('invalid', .01)):
            observations = copy.deepcopy(original)
            for destination in range(world):
                key = (0, destination * 6)
                if invalid is None:
                    del observations[key]
                else:
                    observations[key] = invalid
            with self.subTest(invalid=invalid):
                maps, result = fit_order(rows, observations, world, window)
                self.assertEqual(maps, [list(range(12))] * 2)
                self.assertTrue(all(g['status'] == 'incomplete_training_identity' for g in result['groups']))
                json.dumps(result, allow_nan=False)
        _, result = fit_order(*fixture([1e308, -1e308, 1e308, -1e308], uncertainty=0))
        json.dumps(result, allow_nan=False)

    def test_heldout_does_not_choose_or_reject_mapping(self):
        rows, original, world, window = fixture([0, 0, 10, 10, 8, 8])
        first_maps, first = fit_order(rows, original, world, window)
        changed = copy.deepcopy(original)
        for key in list(changed):
            if key[0] == 2:
                changed[key] = ([1000 - key[1] * 3, key[1] * 555], 100)
        second_maps, second = fit_order(rows, changed, world, window)
        self.assertEqual(first_maps, second_maps)
        self.assertNotEqual([g['heldout'] for g in first['groups']], [g['heldout'] for g in second['groups']])
        for result in (first, second):
            for group in result['groups']:
                del group['heldout']
        self.assertEqual(first, second)
        no_heldout = {key: value for key, value in original.items() if key[0] != 2}
        maps, result = fit_order(rows, no_heldout, world, window)
        self.assertEqual(maps, first_maps)
        self.assertTrue(all(not g['heldout']['available'] for g in result['groups']))

    def test_source_destination_window_bijection_offsets_and_determinism(self):
        rows, observations, world, window = fixture([0, 0, 10, 10, 8, 8] * 2, world=4, window=6)
        maps, result = fit_order(rows, observations, world, window)
        self.assertEqual((maps, result), fit_order(rows[::-1], observations, world, window))
        for source, mapping in enumerate(maps):
            rr = [row for row in rows if row['rank'] == source]
            self.assertEqual(sorted(mapping), list(range(48)))
            for slot, original in enumerate(mapping):
                self.assertEqual(rr[slot]['destination'], rr[original]['destination'])
                self.assertEqual(slot // (world * window), original // (world * window))
                self.assertLessEqual(abs(slot - original) // world, MAX_DISPLACEMENT)
        shifted = {(p, tile): ([value + tile // 12 * 10000 + p * 100 + source * 250
                               for source, value in enumerate(values)], error)
                   for (p, tile), (values, error) in observations.items()}
        self.assertEqual(maps, fit_order(rows, shifted, world, window)[0])
        # Remove one receiver's window and one source's signal independently.
        del observations[0, 0]
        for key, (values, error) in list(observations.items()):
            observations[key] = (values[:3] + [float(key[1] % 12)], error)
        maps, result = fit_order(rows, observations, world, window)
        self.assertEqual(maps[3], list(range(48)))
        self.assertTrue(any(g['status'] == 'candidate' for g in result['groups']))
        self.assertTrue(all(g['status'] == 'incomplete_training_identity' for g in result['groups']
                            if g['destination'] == 0 and g['window_index'] == 0))

    def test_invalid_config_and_mapping_are_rejected(self):
        args = fixture([0, 0, 10, 10, 8, 8])
        for distance in (-1, True, 1.5):
            with self.assertRaises(ValueError):
                fit_order(*args, max_displacement=distance)
        for window in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                fit_order(*args[:3], window)
        rows = copy.deepcopy(args[0])
        rows[0]['index'] = 1
        with self.assertRaises(ValueError):
            fit_order(rows, *args[1:])


if __name__ == '__main__':
    unittest.main()
