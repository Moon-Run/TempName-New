"""CPU checks for fitting and consuming the new order with unchanged codec/mask."""
import contextlib
import io
from pathlib import Path
import unittest
from unittest import mock

import ablation4
import ablation3_plan
import plan
import scope_build
import scope_plan
from common import REPO, validate_entry
from scope_protocol import digest
from test_protocol import fixture_config, mapping, reports


class ArrivalIntegrationTests(unittest.TestCase):
    def test_legacy_default_is_unchanged_and_new_order_preserves_mask(self):
        for world in (4, 8):
            config = fixture_config(world)
            rows, samples = mapping(config, 'interleaved'), reports(config)
            legacy = plan.fit(samples, rows, config, 'interleaved')
            self.assertEqual(legacy, plan.fit(samples, rows, config, 'interleaved', arrival_order='legacy_tail'))
            self.assertNotIn('arrival_order', legacy)
            candidate = plan.fit(samples, rows, config, 'interleaved', arrival_order='tail_residual')
            self.assertEqual(candidate['mask'], legacy['mask'])
            self.assertEqual(candidate['base_coords'], legacy['base_coords'])
            self.assertEqual(candidate['maps'], [list(range(candidate['tiles'])) for _ in range(world)])
            self.assertTrue(candidate['arrival_order']['identity_fallback'])
            validate_entry(candidate, config)
            with self.assertRaisesRegex(ValueError, 'Unknown arrival order'):
                plan.fit(samples, rows, config, 'interleaved', arrival_order='typo')

    def test_full_shape_identity_produces_same_device_code_for_a_and_b(self):
        parent = dict(roles={role: scope_plan.cpu_fixture(role) for role in ('attn', 'mlp')})
        for role, record in parent['roles'].items():
            config = record['config']
            samples = reports(config)
            for sample in samples:
                sample['shapes'][0]['K_global'] = record['entry']['K_global']
            entry = plan.fit(samples, mapping(config, 'interleaved'), config, 'interleaved',
                             record['entry']['shape'], arrival_order='tail_residual')
            record['entry'] = entry
            record['entry_sha256'] = digest(entry)
        a = ablation3_plan.derive(parent, 'interleaved_all_attn_mlp')
        b = ablation3_plan.derive(parent, 'interleaved_arrival_all_attn_mlp')
        ae = {r: p['entry'] for r, p in a['roles'].items()}
        be = {r: p['entry'] for r, p in b['roles'].items()}
        self.assertEqual(ae, be)
        source = (REPO/scope_build.SWIZZLE).read_text()
        self.assertEqual(scope_build.swizzle(source, ae), scope_build.swizzle(source, be))
        self.assertEqual(scope_build.tables(ae), scope_build.tables(be))

    def test_fitter_keeps_heldout_out_of_order_and_mask(self):
        config = fixture_config()
        a = plan.fit(reports(config), mapping(config, 'interleaved'), config, 'interleaved',
                     arrival_order='tail_residual')
        b = plan.fit(reports(config, heldout_shift=1), mapping(config, 'interleaved'), config, 'interleaved',
                     arrival_order='tail_residual')
        for field in ('maps', 'coords', 'mask'):
            self.assertEqual(a[field], b[field])

    def test_joint_order_reaches_full_shape_generated_tables_without_changing_mask(self):
        entries = {}
        for role in ('attn', 'mlp'):
            record = scope_plan.cpu_fixture(role)
            config = record['config']
            samples = reports(config)
            for sample in samples:
                shape = sample['shapes'][0]
                shape['K_global'] = record['entry']['K_global']
                # Two equally late contributions require a joint move. All
                # displaced tiles are included in the candidate's tail cost.
                for observation in shape['samples']:
                    if observation['mode'] == 'receiver_sparse':
                        late_offsets = (2, 3) if role == 'attn' else (6, 7)
                        value = (1 if observation['offset'] < 2 else
                                 11 if observation['offset'] in late_offsets else 9)
                        for fragment in observation['receiver']:
                            fragment['observed_ns'] = 1000000 + value * 1000 + fragment['fragment']
            rows = mapping(config, 'interleaved')
            legacy = plan.fit(samples, rows, config, 'interleaved', record['entry']['shape'])
            single = plan.fit(samples, rows, config, 'interleaved', record['entry']['shape'],
                              arrival_order='tail_residual')
            candidate = plan.fit(samples, rows, config, 'interleaved', record['entry']['shape'],
                                 arrival_order='tail_assignment')
            self.assertTrue(single['arrival_order']['identity_fallback'])
            self.assertFalse(candidate['arrival_order']['identity_fallback'])
            self.assertEqual(candidate['mask'], legacy['mask'])
            self.assertEqual(candidate['base_coords'], legacy['base_coords'])
            validate_entry(candidate, config, record['entry']['shape'])
            entries[role] = candidate
        self.assertNotEqual(entries['attn']['maps'], entries['mlp']['maps'])
        source = (REPO/scope_build.SWIZZLE).read_text()
        device = scope_build.swizzle(source, entries)
        host = scope_build.tables(entries)
        self.assertIn('phase6_device_attn_lookup', device)
        self.assertIn('phase6_device_mlp_lookup', device)
        self.assertIn('phase6_host_attn_lookup', host)
        self.assertIn('phase6_host_mlp_lookup', host)

    def test_joint_fit_heldout_does_not_select_order_or_mask(self):
        config = fixture_config()
        first = plan.fit(reports(config), mapping(config, 'interleaved'), config, 'interleaved',
                         arrival_order='tail_assignment')
        altered = plan.fit(reports(config, heldout_shift=1), mapping(config, 'interleaved'),
                           config, 'interleaved', arrival_order='tail_assignment')
        for field in ('maps', 'coords', 'mask'):
            self.assertEqual(first[field], altered[field])

    def test_ablation4_fit_keeps_legacy_default_and_accepts_explicit_candidate(self):
        for option, expected in (([], 'legacy_tail'), (['--arrival-order', 'legacy_tail'], 'legacy_tail'),
                                 (['--arrival-order', 'tail_residual'], 'tail_residual'),
                                 (['--arrival-order', 'tail_assignment'], 'tail_assignment')):
            with mock.patch('sys.argv', ['ablation4.py', 'fit', '/calibration', '/new-plan', *option]), \
                 mock.patch.object(ablation4.scope_calibrate, 'verify', return_value={'communication': {}}), \
                 mock.patch.object(ablation4.scope_calibrate, 'calibration_records', return_value=[]), \
                 mock.patch.object(ablation4.scope_calibrate, 'fit') as fit, contextlib.redirect_stdout(io.StringIO()):
                ablation4.main()
                fit.assert_called_once_with(Path('/calibration'), Path('/new-plan'), arrival_order=expected)


if __name__ == '__main__':
    unittest.main()
