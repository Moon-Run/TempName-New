"""Configured margins preserve frozen v2 behavior and expose exploratory risk."""
import copy
import hashlib
import json
import math
import unittest

import arrival_assignment as assignment
from arrival_tail import _predicted, _samples
from test_arrival_assignment import fixture


def options(multiplier):
    return {'uncertainty_multiplier': multiplier}


def fingerprint(value):
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(encoded.encode()).hexdigest()


class ArrivalAssignmentMarginTests(unittest.TestCase):
    def test_omitted_options_preserve_complete_frozen_v2_output(self):
        # Captured before the configured-margin change, including all mapping,
        # diagnostic and limitation fields. Archived v2 plans hash these values.
        frozen = {
            .01: '89a77231b0fdc91eca6d6abf7bee583d3c995a44aa150c4710fd3978a42fa018',
            .2: 'dd4aa6ca87e67ffe6d5a2780fedb11c4da17443dc905a2d2da476d11bc4290bf',
        }
        for error, expected in frozen.items():
            args = fixture([0, 0, 10, 10, 8, 8], uncertainty=error)
            with self.subTest(error=error):
                result = assignment.fit_order(*args)
                self.assertEqual(fingerprint(result), expected)
                self.assertEqual(result, assignment.fit_order(*args, options=None))
                diagnostics = result[1]
                self.assertEqual(diagnostics['method'], assignment.METHOD)
                self.assertEqual(diagnostics['uncertainty_multiplier'], 6.)
                self.assertNotIn('arrival_options', diagnostics)
                self.assertNotIn('exploratory', diagnostics)
                self.assertNotIn('conditional_error_multiplier', diagnostics)
                self.assertNotIn('accepted_windows_with_positive_conservative_bound', diagnostics)

    def test_explicit_six_keeps_mapping_but_records_configured_method(self):
        for error in (.01, .2):
            args = fixture([0, 0, 10, 10, 8, 8], uncertainty=error)
            legacy_maps, legacy = assignment.fit_order(*args)
            configured_maps, configured = assignment.fit_order(*args, options=options(6))
            self.assertEqual(legacy_maps, configured_maps)
            self.assertEqual(configured['method'], assignment.CONFIGURED_METHOD)
            self.assertEqual(configured['method'], 'receiver-tail-assignment-v3')
            self.assertEqual(configured['arrival_options'], options(6.))
            self.assertEqual(configured['uncertainty_multiplier'], 6.)
            self.assertEqual(configured['conditional_error_multiplier'], 4.)
            self.assertFalse(configured['exploratory'])
            self.assertFalse(configured['gpu_validated'])
            self.assertEqual(legacy['accepted_windows'], configured['accepted_windows'])
            for old, new in zip(legacy['groups'], configured['groups']):
                self.assertEqual(old['status'], new['status'])
                self.assertEqual(old['training'], new['training'])
                self.assertEqual(old['heldout'], new['heldout'])
                if new['candidate'] is not None:
                    extra = copy.deepcopy(new['candidate'])
                    lower = extra.pop('conservative_gain_lower_bound_us')
                    self.assertEqual(extra, old['candidate'])
                    for gain, bound in zip(extra['gain_us'], lower):
                        self.assertAlmostEqual(bound, gain - 4 * error)

    def test_alpha_one_admits_exploration_without_claiming_positive_error_bound(self):
        args = fixture([0, 0, 10, 10, 8, 8], uncertainty=.4)
        before = copy.deepcopy(args)
        self.assertTrue(assignment.fit_order(*args)[1]['identity_fallback'])
        self.assertTrue(assignment.fit_order(*args, options=options(4))[1]['identity_fallback'])
        _, result = assignment.fit_order(*args, options=options(1))
        self.assertEqual(before, args)
        self.assertFalse(result['identity_fallback'])
        self.assertTrue(result['exploratory'])
        self.assertFalse(result['gpu_validated'])
        self.assertEqual(result['accepted_windows'], 4)
        self.assertEqual(result['accepted_windows_with_positive_conservative_bound'], 0)
        for group in result['groups']:
            candidate = group['candidate']
            self.assertTrue(all(gain > margin for gain, margin in
                                zip(candidate['gain_us'], candidate['required_margin_us'])))
            self.assertTrue(all(bound < 0 for bound in candidate['conservative_gain_lower_bound_us']))
            self.assertEqual(candidate['required_margin_us'], [.4, .4])
            self.assertEqual([sample['max_uncertainty_us'] for sample in group['training']], [.4, .4])
        json.dumps(result, allow_nan=False)

    def test_four_epsilon_acceptance_has_positive_conditional_bound(self):
        args = fixture([0, 0, 10, 10, 8, 8], uncertainty=.2)
        self.assertTrue(assignment.fit_order(*args)[1]['identity_fallback'])
        _, result = assignment.fit_order(*args, options=options(4))
        self.assertFalse(result['identity_fallback'])
        self.assertFalse(result['exploratory'])
        self.assertEqual(result['accepted_windows_with_positive_conservative_bound'], result['accepted_windows'])
        for group in result['groups']:
            for gain, bound in zip(group['candidate']['gain_us'],
                                   group['candidate']['conservative_gain_lower_bound_us']):
                self.assertAlmostEqual(bound, gain - .8)
                self.assertGreater(bound, 0)

    def test_strict_margin_rejects_equality_and_admits_resolved_excess(self):
        args = fixture([0, 0, 10, 10, 8, 8], uncertainty=.2)
        # Best possible gain is 1 us. Equality at 5*.2 must not be accepted.
        _, equal = assignment.fit_order(*args, options=options(5))
        self.assertTrue(equal['identity_fallback'])
        self.assertEqual(equal['accepted_windows'], 0)
        self.assertEqual(equal['accepted_windows_with_positive_conservative_bound'], 0)
        _, smaller = assignment.fit_order(*args, options=options(4.999))
        self.assertFalse(smaller['identity_fallback'])
        # The newly justified 4*error boundary itself is also strict.
        _, equal_four = assignment.fit_order(
            *fixture([0, 0, 10, 10, 8, 8], uncertainty=.25), options=options(4))
        self.assertTrue(equal_four['identity_fallback'])

    def test_heldout_cannot_select_margin_order_or_conditional_bound(self):
        rows, observations, world, window = fixture([0, 0, 10, 10, 8, 8], uncertainty=.4)
        altered = copy.deepcopy(observations)
        for key in list(altered):
            if key[0] == 2:
                altered[key] = ([1000 - key[1] * 7, key[1] * 555], 100.)
        first_maps, first = assignment.fit_order(rows, observations, world, window, options=options(1))
        second_maps, second = assignment.fit_order(rows, altered, world, window, options=options(1))
        self.assertEqual(first_maps, second_maps)
        self.assertNotEqual([g['heldout'] for g in first['groups']], [g['heldout'] for g in second['groups']])
        for result in (first, second):
            for group in result['groups']:
                self.assertNotIn('conservative_gain_lower_bound_us', group['heldout'])
                del group['heldout']
        self.assertEqual(first, second)
        training_only = {key: value for key, value in observations.items() if key[0] in (0, 1)}
        third_maps, third = assignment.fit_order(rows, training_only, world, window, options=options(1))
        self.assertEqual(first_maps, third_maps)
        for group in third['groups']:
            self.assertFalse(group['heldout']['available'])
            del group['heldout']
        self.assertEqual(first, third)

    def test_options_are_validated_normalized_and_not_mutated(self):
        self.assertIsNone(assignment.normalize_options(None))
        for multiplier in (1, 1.5, 4, 6.):
            supplied = options(multiplier)
            before = copy.deepcopy(supplied)
            normalized = assignment.normalize_options(supplied)
            self.assertEqual(normalized, options(float(multiplier)))
            self.assertIs(type(normalized['uncertainty_multiplier']), float)
            self.assertEqual(supplied, before)
            self.assertIsNot(supplied, normalized)
        bad = [True, 2, '1', [], {}, {'other': 1},
               {'uncertainty_multiplier': 1, 'other': 2}]
        bad += [options(value) for value in (True, False, None, '2', math.nan, math.inf,
                                             -math.inf, 0, .9999, 6.0001, -1, 2j)]
        args = fixture([0, 0, 10, 10, 8, 8])
        for supplied in bad:
            with self.subTest(options=supplied):
                with self.assertRaises(ValueError):
                    assignment.normalize_options(supplied)
                with self.assertRaises(ValueError):
                    assignment.fit_order(*args, options=supplied)

    def test_correlated_pava_gain_error_can_exceed_three_epsilon(self):
        # Uniform error bound proof: PAVA error <= e; a shifted prediction
        # error <= 3e; identity maximum error <= e. Shared observations do not
        # justify replacing the resulting 4e gain bound by 2e or 3e.
        values = [0., 15., 30., 29.] + [17.625] * 8
        error = .01
        perturbation = [0., -1., 1., -1.] + [1.] * 8
        changed = [value + error * sign for value, sign in zip(values, perturbation)]
        order = [2, 3, 0, 1] + list(range(4, 12))

        def sample(arrivals):
            observations = {(0, tile): ([value, value], error) for tile, value in enumerate(arrivals)}
            return _samples(observations, 0, list(range(12)), 0, 2)

        original, perturbed = sample(values), sample(changed)
        self.assertEqual(original['baseline'], [0., 15.] + [20.] * 10)
        baseline_error = max(abs(x - y) for x, y in zip(original['baseline'], perturbed['baseline']))
        self.assertLessEqual(baseline_error, error + 1e-12)
        original_gain = max(values) - max(_predicted(original, order))
        perturbed_gain = max(changed) - max(_predicted(perturbed, order))
        gain_error = abs(original_gain - perturbed_gain)
        self.assertAlmostEqual(original_gain, 6.)
        self.assertAlmostEqual(gain_error / error, 3.8)
        self.assertGreater(gain_error, 3 * error)
        self.assertLessEqual(gain_error, 4 * error + 1e-12)


if __name__ == '__main__':
    unittest.main()
