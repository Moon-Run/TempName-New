"""CPU invariants for the conservative receiver-tail candidate planner."""
import copy
import json
import math
import unittest

from arrival_tail import fit_order, MAX_SWAPS_PER_WINDOW


def fixture(values, world=2, uncertainty=0.01, window=None):
    """An interleaved grid; each destination has len(values) physical tiles."""
    per = len(values)
    rows = []
    for source in range(world):
        for i in range(per * world):
            destination = (i % world + source) % world
            rows.append(dict(rank=source, index=i, m=destination, n=i // world,
                             destination=destination))
    observations = {(p, destination * per + within):
                    ([float(value) + source * 0.5 + p * 0.1 for source in range(world)], uncertainty)
                    for p in (0, 1, 2) for destination in range(world)
                    for within, value in enumerate(values)}
    return rows, observations, world, per if window is None else window


class ArrivalTailTests(unittest.TestCase):
    def test_pure_slot_delay_is_identity(self):
        for values in ([1, 2, 3, 4, 5, 6], [1, 1, 1, 10, 10, 25], [3] * 6):
            with self.subTest(values=values):
                maps, result = fit_order(*fixture(values))
                self.assertTrue(result['identity_fallback'])
                self.assertEqual(result['accepted_swaps'], 0)
                self.assertEqual(maps, [list(range(12))] * 2)

    def test_stable_residual_tail_moves_earlier_with_two_pass_gain(self):
        rows, observations, world, window = fixture([1, 2, 3, 20, 5, 6])
        before = copy.deepcopy((rows, observations))
        maps, result = fit_order(rows, observations, world, window)
        self.assertFalse(result['identity_fallback'])
        self.assertGreater(result['accepted_swaps'], 0)
        self.assertEqual((rows, observations), before)
        for group in result['groups']:
            self.assertEqual(group['status'], 'candidate')
            self.assertLessEqual(len(group['accepted_swaps']), MAX_SWAPS_PER_WINDOW)
            for swap in group['accepted_swaps']:
                self.assertTrue(all(gain > margin for gain, margin in
                                    zip(swap['gain_us'], swap['required_margin_us'])))
            for train in group['training']:
                self.assertLess(train['after_max_us'], train['before_max_us'])
                self.assertEqual(train['after_max_us'], max(train['predicted_arrivals_us']))
            # The observed straggler is the fourth physical tile in this group.
            original = group['slots'][3]
            self.assertLess(maps[group['source']].index(original), original)
        json.dumps(result, allow_nan=False)

    def test_displaced_tile_is_counted_and_equal_tail_is_rejected(self):
        # Two tied tails: advancing only one does not improve the worst arrival.
        maps, result = fit_order(*fixture([1, 2, 20, 20, 5, 6]))
        self.assertTrue(result['identity_fallback'])
        self.assertEqual(maps, [list(range(12))] * 2)
        # Within a PAVA plateau, moving a later residual to the same fitted
        # baseline cannot help even if it is much later than its immediate peer.
        _, result = fit_order(*fixture([20, 1]))
        self.assertTrue(result['identity_fallback'])

    def test_both_training_passes_must_resolve_gain(self):
        args = fixture([1, 2, 3, 20, 5, 6])
        rows, obs, world, window = args
        for (p, tile), (arrivals, error) in list(obs.items()):
            if p == 1:
                obs[p, tile] = ([float(tile % 6) + source for source in range(world)], error)
        maps, result = fit_order(rows, obs, world, window)
        self.assertTrue(result['identity_fallback'])
        self.assertEqual(maps, [list(range(12))] * 2)

    def test_source_and_receiver_labels_are_not_shared(self):
        rows, obs, world, window = fixture([1, 2, 3, 4, 5, 6])
        for p in (0, 1, 2):
            arrivals, error = obs[p, 3]
            obs[p, 3] = ([20.0 + p * 0.1, arrivals[1]], error)
        maps, result = fit_order(rows, obs, world, window)
        self.assertNotEqual(maps[0], list(range(12)))
        self.assertEqual(maps[1], list(range(12)))
        self.assertEqual([(g['source'], g['destination']) for g in result['groups']
                          if g['accepted_swaps']], [(0, 0)])

    def test_missing_invalid_and_noisy_training_leave_identity(self):
        rows, original, world, window = fixture([1, 2, 3, 20, 5, 6])
        for invalid in (None, ([math.nan, 1], 0.01), ([1, 2], -1), ([1, 2], math.inf),
                        ([1], 0.01), ('invalid', 0.01)):
            obs = copy.deepcopy(original)
            for destination in range(world):
                key = (0, destination * 6)
                if invalid is None:
                    del obs[key]
                else:
                    obs[key] = invalid
            with self.subTest(invalid=invalid):
                maps, result = fit_order(rows, obs, world, window)
                self.assertTrue(result['identity_fallback'])
                self.assertTrue(all(g['status'] == 'incomplete_training_identity' for g in result['groups']))
                self.assertEqual(maps, [list(range(12))] * 2)
        _, result = fit_order(*fixture([1, 2, 3, 20, 5, 6], uncertainty=10))
        self.assertTrue(result['identity_fallback'])
        # The best modeled gain is between 8 and 12 us: a 2 us measurement
        # uncertainty passes a naive 4*error gate but fails the 6*error bound
        # needed when both current and proposed predictions were shifted.
        _, result = fit_order(*fixture([1, 2, 3, 20, 5, 6], uncertainty=2))
        self.assertTrue(result['identity_fallback'])

    def test_heldout_is_diagnostic_only_and_never_changes_training(self):
        rows, obs, world, window = fixture([1, 2, 3, 20, 5, 6])
        first_maps, first = fit_order(rows, obs, world, window)
        altered = copy.deepcopy(obs)
        for p, tile in list(altered):
            if p == 2:
                altered[p, tile] = ([1000.0 - tile * 3, tile * 555.0], 100.0)
        other_maps, other = fit_order(rows, altered, world, window)
        self.assertEqual(first_maps, other_maps)
        self.assertNotEqual([g['heldout'] for g in first['groups']], [g['heldout'] for g in other['groups']])
        for result in (first, other):
            for group in result['groups']:
                del group['heldout']
        self.assertEqual(first, other)
        no_heldout = {key: value for key, value in obs.items() if key[0] != 2}
        maps, result = fit_order(rows, no_heldout, world, window)
        self.assertEqual(maps, first_maps)
        self.assertTrue(all(not g['heldout']['available'] for g in result['groups']))

    def test_bijection_destination_window_determinism_and_receiver_offset(self):
        rows, obs, world, window = fixture([1, 2, 3, 20, 5, 6] * 2, world=4, window=6)
        maps, result = fit_order(rows, obs, world, window)
        by_source = [[r for r in rows if r['rank'] == source] for source in range(world)]
        for source, mapping in enumerate(maps):
            self.assertEqual(sorted(mapping), list(range(48)))
            for i, j in enumerate(mapping):
                self.assertEqual(by_source[source][i]['destination'], by_source[source][j]['destination'])
                self.assertEqual(i // (world * window), j // (world * window))
        self.assertEqual((maps, result), fit_order(rows[::-1], obs, world, window))
        # Arbitrarily shift each receiver/pass/source clock origin. Since fits
        # and decisions are local, the mapping must be unchanged.
        shifted = {(p, t): ([a + (t // 12) * 10000 + p * 100 + source * 250
                            for source, a in enumerate(arrivals)], error)
                   for (p, t), (arrivals, error) in obs.items()}
        self.assertEqual(maps, fit_order(rows, shifted, world, window)[0])

    def test_incomplete_window_does_not_disable_other_valid_window(self):
        rows, obs, world, window = fixture([1, 2, 3, 20, 5, 6] * 2, window=6)
        del obs[0, 0]
        maps, result = fit_order(rows, obs, world, window)
        self.assertFalse(result['identity_fallback'])
        for group in result['groups']:
            if group['destination'] == 0 and group['window_index'] == 0:
                self.assertEqual(group['status'], 'incomplete_training_identity')
                self.assertTrue(all(maps[group['source']][i] == i for i in group['slots']))
            else:
                self.assertEqual(group['status'], 'candidate')

    def test_invalid_mapping_or_window_is_rejected(self):
        rows, obs, world, window = fixture([1, 2, 3, 20, 5, 6])
        for bad_window in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                fit_order(rows, obs, world, bad_window)
        for mutate in ('duplicate', 'rank', 'physical', 'destination'):
            broken = copy.deepcopy(rows)
            if mutate == 'duplicate': broken[0]['index'] = 1
            if mutate == 'rank': broken[0]['rank'] = world
            if mutate == 'physical': broken[0]['n'] = 1
            if mutate == 'destination': broken[0]['destination'] = 1
            with self.subTest(mutate=mutate), self.assertRaises(ValueError):
                fit_order(broken, obs, world, window)


if __name__ == '__main__':
    unittest.main()
