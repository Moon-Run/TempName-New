"""CPU admission fixtures for the user-approved D8 reference error budget."""
import copy
import unittest
from unittest import mock

import mav2_validation as validation


def evidence():
    # Deliberately synthetic in-memory fixtures; no fixture is saved as GPU evidence.
    nonfinite = [dict(value=v, rejected=True, separate_cuda_context=True) for v in ('nan', 'inf')]
    common = dict(passed=True, synthetic=False, rank=3, retained_outputs_unchanged=True,
                  quantized_remote_fraction=1.0, bytes_per_group=136, nonfinite_checks=nonfinite)
    selective = dict(common, selector=validation.SELECTOR, cases=[])
    direct = dict(common, policy='d8', quality_selection=False, h8_fallback_allowed=False, cases=[])
    for case in validation.CASES:
        shape = [256, 4096] if case == 'full_shape' else [8, 128]
        total = 7*shape[0]*shape[1]//128
        diagnostics = dict(total=total, d8=total, h8=0, nonfinite=0, invalid_tags=0,
                           selector=validation.SELECTOR, bytes_per_group=136)
        selective_diag = dict(diagnostics, d8=1, h8=total-1)
        row = dict(case=case, passed=True, finite=True, shape=shape, relative_l2=.025,
                   protocol_relative_l2=0., packet_relative_l2=0., ordered_bf16_fold_equal=True)
        selective['cases'].append(dict(row, diagnostics=selective_diag, forced_h8_packet_equal=True,
            forced_h8_decode_equal=True, forced_h8_fold_equal=True, forced_d8_relative_l2=0.,
            forced_d8_reference_budget=.008))
        direct['cases'].append(dict(row, diagnostics=diagnostics, formal_equals_diagnostic=True,
            policy='d8', quality_selection=False, h8_fallback_allowed=False, budget_vs_reference=.008,
            rounding_midpoint=(dict(scale_exact=True, payload_exact=True, reference_equal=True)
                               if case == 'rounding_midpoint' else None)))
    return dict(selective_transform=selective, forced_direct=direct)


class ValidationBudgetTests(unittest.TestCase):
    def admit(self, record):
        # The unchanged basic codec has its own validation suite. Exercise only
        # the real MAV2/D8 evidence verifier, including its independent limits.
        with mock.patch.object(validation.baseline, 'verify'):
            validation.verify(record, {'synthetic': 'CPU fixture'}, 3)

    def test_observed_05139_percent_error_is_allowed(self):
        self.assertEqual(validation.D8_REFERENCE_L2_LIMIT, .008)
        record = evidence()
        observed = .005138692005978244
        record['selective_transform']['cases'][1]['forced_d8_relative_l2'] = observed
        record['forced_direct']['cases'][1]['protocol_relative_l2'] = observed
        self.admit(record)

    def test_both_reference_checks_use_inclusive_point_eight_percent(self):
        for section, field in (('selective_transform', 'forced_d8_relative_l2'),
                               ('forced_direct', 'protocol_relative_l2')):
            with self.subTest(section=section):
                record = evidence()
                record[section]['cases'][1][field] = .008
                self.admit(record)
                for value in (.008001, float('inf'), float('nan'), -.001):
                    changed = copy.deepcopy(record)
                    changed[section]['cases'][1][field] = value
                    with self.assertRaises(ValueError):
                        self.admit(changed)

    def test_packet_and_input_budgets_are_not_relaxed(self):
        for section in ('selective_transform', 'forced_direct'):
            for field, error in (('packet_relative_l2', .0051), ('relative_l2', .0501)):
                with self.subTest(section=section, field=field):
                    record = evidence()
                    record[section]['cases'][1][field] = error
                    with self.assertRaises(ValueError):
                        self.admit(record)

    def test_stale_reference_budget_metadata_is_rejected(self):
        for section, field in (('selective_transform', 'forced_d8_reference_budget'),
                               ('forced_direct', 'budget_vs_reference')):
            record = evidence()
            record[section]['cases'][1][field] = .005
            with self.subTest(section=section), self.assertRaises(ValueError):
                self.admit(record)

    def test_midpoint_regression_requires_exact_device_evidence(self):
        for field in ('scale_exact', 'payload_exact', 'reference_equal'):
            record = evidence()
            row = next(r for r in record['forced_direct']['cases'] if r['case'] == 'rounding_midpoint')
            row['rounding_midpoint'][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.admit(record)


if __name__ == '__main__':
    unittest.main()
