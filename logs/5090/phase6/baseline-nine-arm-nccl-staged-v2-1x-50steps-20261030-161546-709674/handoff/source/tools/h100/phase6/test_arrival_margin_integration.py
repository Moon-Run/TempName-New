"""CPU integration checks for explicit exploratory margins and unchanged masks."""
import contextlib
import copy
import io
from pathlib import Path
import unittest
from unittest import mock

import ablation4
import arrival_assignment
import plan
import scope_build
import scope_plan
from common import validate_entry
from test_protocol import mapping, reports


def samples_for_role(role):
    record = scope_plan.cpu_fixture(role)
    config = record['config']
    samples = reports(config)
    late = (2, 3) if role == 'attn' else (6, 7)
    for report in samples:
        shape = report['shapes'][0]
        shape['K_global'] = record['entry']['K_global']
        for observation in shape['samples']:
            if observation['mode'] == 'receiver_sparse':
                observation['max_poll_cycle_ns'] = 600  # epsilon remains 1.2 us.
                value = 1 if observation['offset'] < 2 else 11 if observation['offset'] in late else 9
                for fragment in observation['receiver']:
                    fragment['observed_ns'] = 1000000 + value * 1000 + fragment['fragment']
    return record, config, samples


class MarginIntegrationTests(unittest.TestCase):
    def test_margin_reaches_both_shape_tables_without_rescaling_errors_or_changing_masks(self):
        entries = {}
        for role in ('attn', 'mlp'):
            record, config, samples = samples_for_role(role)
            rows = mapping(config, 'interleaved')
            shape = record['entry']['shape']
            legacy = plan.fit(samples, rows, config, 'interleaved', shape, arrival_order='tail_assignment')
            candidate = plan.fit(samples, rows, config, 'interleaved', shape,
                arrival_order='tail_assignment', arrival_options={'uncertainty_multiplier': 1})
            self.assertTrue(legacy['arrival_order']['identity_fallback'])
            self.assertFalse(candidate['arrival_order']['identity_fallback'])
            self.assertEqual(candidate['arrival_order']['method'], arrival_assignment.CONFIGURED_METHOD)
            self.assertEqual(candidate['mask'], legacy['mask'])
            self.assertEqual(candidate['base_coords'], legacy['base_coords'])
            self.assertEqual(candidate['selected_per_source'], legacy['selected_per_source'])
            for before, after in zip(legacy['arrival_order']['groups'], candidate['arrival_order']['groups']):
                for a, b in zip(before['training'], after['training']):
                    self.assertEqual(a['arrivals_us'], b['arrivals_us'])
                    self.assertEqual(a['max_uncertainty_us'], b['max_uncertainty_us'])
                    self.assertEqual(b['max_uncertainty_us'], 1.2)
            validate_entry(candidate, config, shape)
            entries[role] = candidate
        self.assertNotEqual(entries['attn']['maps'], entries['mlp']['maps'])
        self.assertIn('phase6_host_attn_lookup', scope_build.tables(entries))
        self.assertIn('phase6_device_mlp_lookup', scope_build.tables(entries, device=True))

    def test_heldout_changes_neither_exploratory_permutation_nor_physical_mask(self):
        record, config, samples = samples_for_role('attn')
        altered = copy.deepcopy(samples)
        for report in altered:
            for sample in report['shapes'][0]['samples']:
                if sample['mode'] == 'receiver_sparse' and sample['repeat'] >= 128:
                    for fragment in sample['receiver']:
                        fragment['observed_ns'] += (fragment['source'] + 1) * 100000
        kwargs = dict(arrival_order='tail_assignment', arrival_options={'uncertainty_multiplier': 1})
        args = (mapping(config, 'interleaved'), config, 'interleaved', record['entry']['shape'])
        original = plan.fit(samples, *args, **kwargs)
        other = plan.fit(altered, *args, **kwargs)
        for field in ('maps', 'coords', 'mask'):
            self.assertEqual(original[field], other[field])
        self.assertEqual(original['arrival_order']['arrival_options'], other['arrival_order']['arrival_options'])

    def test_options_cannot_silently_change_legacy_or_residual_methods(self):
        record, config, samples = samples_for_role('attn')
        for method in ('legacy_tail', 'tail_residual'):
            with self.subTest(method=method), self.assertRaisesRegex(ValueError, 'require tail_assignment'):
                plan.fit(samples, mapping(config, 'interleaved'), config, 'interleaved',
                         arrival_order=method, arrival_options={'uncertainty_multiplier': 1})

    def test_cli_explicit_margin_is_forwarded_and_wrong_method_rejected(self):
        with mock.patch('sys.argv', ['ablation4.py', 'fit', '/calibration', '/new-plan',
                                    '--arrival-order', 'tail_assignment', '--arrival-margin-multiplier', '1']), \
             mock.patch.object(ablation4.scope_calibrate, 'verify', return_value={'communication': {}}), \
             mock.patch.object(ablation4.scope_calibrate, 'calibration_records', return_value=[]), \
             mock.patch.object(ablation4.scope_calibrate, 'fit') as fit, \
             contextlib.redirect_stdout(io.StringIO()):
            ablation4.main()
            fit.assert_called_once_with(Path('/calibration'), Path('/new-plan'),
                arrival_order='tail_assignment', arrival_options={'uncertainty_multiplier': 1.0})
        for value in ('0', 'nan', '7'):
            with mock.patch('sys.argv', ['ablation4.py', 'fit', '/missing', '/new-plan',
                    '--arrival-order', 'tail_assignment', '--arrival-margin-multiplier', value]), \
                 mock.patch.object(ablation4.scope_calibrate, 'verify', side_effect=AssertionError('Must validate margin first')), \
                 self.assertRaises(ValueError):
                ablation4.main()


if __name__ == '__main__':
    unittest.main()
