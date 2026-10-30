"""Frozen exploratory margins, legacy compatibility, and end-to-end evidence seals."""
import contextlib
import copy
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import ablation4_build
import ablation7_new as cli
import ablation7_new_build as builder
import ablation7_new_plan as planner
import ablation7_new_prepare as preparation
import ablation7_new_protocol as routing
import ablation7_new_report as reporter
import arrival_assignment
import arrival_probe
import detach_ablation7_new as batch
import scope_plan
from common import read, sha, write
import test_ablation7_report as report_fixtures


OPTIONS = {'uncertainty_multiplier': 1.0}


def parent_fixture():
    return dict(protocol=scope_plan.PROTOCOL, source_only=True, gpu_calibrated=False,
                roles={role: scope_plan.cpu_fixture(role) for role in routing.SHAPES})


class MarginPlanTests(unittest.TestCase):
    def test_default_programmatic_plan_remains_v2_and_explicit_margin_selects_v3(self):
        parent = parent_fixture()
        legacy = planner.bundle(parent, source_only=True)
        configured = planner.bundle(parent, source_only=True, arrival_options=OPTIONS)
        self.assertNotIn('arrival_options', legacy)
        self.assertEqual(configured['arrival_options'], OPTIONS)
        self.assertEqual(configured['variants'], legacy['variants'])
        self.assertNotEqual(routing.digest(configured), routing.digest(legacy))
        for role in routing.SHAPES:
            old = legacy['orders'][role]['entry']['arrival_order']
            new = configured['orders'][role]['entry']['arrival_order']
            self.assertEqual(old['method'], arrival_assignment.METHOD)
            self.assertEqual(old['uncertainty_multiplier'], 6)
            self.assertNotIn('arrival_options', old)
            self.assertNotIn('exploratory', old)
            self.assertEqual(new['method'], arrival_assignment.CONFIGURED_METHOD)
            self.assertEqual(new['arrival_options'], OPTIONS)
            self.assertTrue(new['exploratory'])
            self.assertEqual(new['conditional_error_multiplier'], 4.0)

    def test_relabelled_margin_or_method_fails_even_if_entry_digest_is_updated(self):
        plans = planner.bundle(parent_fixture(), source_only=True, arrival_options=OPTIONS)
        for change in ('top_margin', 'diagnostic_margin', 'method', 'exploratory', 'bound_count'):
            bad = copy.deepcopy(plans)
            order = bad['orders']['attn']; diagnostic = order['entry']['arrival_order']
            if change == 'top_margin': bad['arrival_options']['uncertainty_multiplier'] = 2.0
            elif change == 'diagnostic_margin': diagnostic['uncertainty_multiplier'] = 6
            elif change == 'method': diagnostic['method'] = arrival_assignment.METHOD
            elif change == 'exploratory': diagnostic['exploratory'] = False
            else: diagnostic['accepted_windows_with_positive_conservative_bound'] = 1
            order['entry_sha256'] = routing.digest(order['entry'])
            with self.subTest(change=change), self.assertRaises(ValueError): planner.validate(bad)
        for value in (None, {}, {'uncertainty_multiplier': 0}, {'uncertainty_multiplier': True}):
            bad = dict(plans, arrival_options=value)
            with self.subTest(value=value), self.assertRaises(ValueError): planner.validate(bad)

    def test_cli_is_explicit_exploratory_while_replay_defaults_are_unchanged(self):
        args = cli.parser().parse_args(['build-prepare', '/new', '--parent-build', '/parent'])
        self.assertEqual(args.arrival_margin_multiplier, 1.0)
        args = batch.arguments(['--out', '/tmp/arrival-margin-preview'])
        command = batch.commands(args)[0][1]
        self.assertEqual(command[command.index('--arrival-margin-multiplier')+1], '1.0')
        self.assertEqual(len([label for label, _ in batch.commands(args) if label.endswith('-controller')]), 4)
        for value in ('0', '7', 'nan', 'inf'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                batch.arguments(['--arrival-margin-multiplier', value])

    def test_legacy_report_metadata_has_exact_old_fields(self):
        plans = planner.bundle(parent_fixture(), source_only=True)
        expected = {'method', 'identity_fallback', 'accepted_windows', 'window_status_counts',
                    'max_displacement', 'training_passes', 'heldout_passes', 'changed_per_rank',
                    'identical_to_A', 'fitted_entry_sha256'}
        for metadata in reporter.arrival_metadata(plans).values():
            self.assertEqual(set(metadata), expected)


class MarginPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ablation7-margin-CPU-')
        cls.root = Path(cls.temp.name)
        cls.parent = ablation4_build.prepare(cls.root/'parent', source_only=True)
        cls.legacy_build = builder.prepare(cls.root/'legacy', cls.parent, source_only=True)
        cls.build = builder.prepare(cls.root/'configured', cls.parent, source_only=True, arrival_options=OPTIONS)
        cls.compare = preparation.prepare(cls.root/'compare', cls.build, '/unused', source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_options_are_frozen_across_build_submission_worker_and_preparation(self):
        self.assertEqual(reporter.verify_inputs(self.compare)['arrival_options'], OPTIONS)
        paths = (self.build/'plans.json', self.build/'ablation7-new-manifest.json',
                 self.build/'improved/manifest.json', self.compare/'submission.json',
                 self.compare/'scripts/config.json', self.compare/'provenance/preparation.json')
        for path in paths:
            self.assertEqual(read(path)['arrival_options'], OPTIONS, str(path))
        summary = batch.order_summary(self.compare)
        self.assertEqual(summary, reporter.arrival_metadata(read(self.build/'plans.json')))
        for metadata in summary.values():
            self.assertEqual(metadata['method'], arrival_assignment.CONFIGURED_METHOD)
            self.assertEqual(metadata['arrival_options'], OPTIONS)
            self.assertTrue(metadata['exploratory'])
            self.assertIsNone(metadata['minimum_conservative_gain_lower_bound_us'])

    def test_reuse_rejects_legacy_and_different_margins_before_execution(self):
        with self.assertRaisesRegex(ValueError, 'Reused build arrival margin'):
            batch.check_build_options(self.legacy_build, OPTIONS)
        with self.assertRaisesRegex(ValueError, 'Reused build arrival margin'):
            batch.check_build_options(self.build, {'uncertainty_multiplier': 2.0})
        self.assertEqual(batch.check_build_options(self.build, OPTIONS)['arrival_options'], OPTIONS)

    def test_report_rejects_worker_margin_drift_with_updated_file_hash(self):
        config_path, info_path = self.compare/'scripts/config.json', self.compare/'submission.json'
        originals = [path.read_bytes() for path in (config_path, info_path)]
        try:
            config, info = read(config_path), read(info_path)
            config['arrival_options']['uncertainty_multiplier'] = 2.0
            write(config_path, config)
            info['files']['scripts/config.json'] = sha(config_path)
            write(info_path, info)
            with self.assertRaisesRegex(ValueError, 'arrival margin'):
                reporter.verify_inputs(self.compare, offline=True)
        finally:
            for path, content in zip((config_path, info_path), originals): path.write_bytes(content)

    def test_build_snapshot_rejects_margin_drift(self):
        record = read(self.build/'ablation7-new-manifest.json')
        record['arrival_options']['uncertainty_multiplier'] = 2.0
        with self.assertRaisesRegex(ValueError, 'arrival margin'):
            builder.validate_snapshots(record, read(self.build/'plans.json'),
                read(self.build/'parent-candidate-manifest.json'), read(self.build/'parent-fused-manifest.json'),
                read(self.build/'improved/manifest.json'))

    def test_probe_signature_includes_margin_and_complete_plan_hash(self):
        info = read(self.compare/'submission.json'); info.update(executable=True, source_only=False)
        with mock.patch.object(reporter, 'verify_inputs', return_value=info):
            inspected = arrival_probe.inspect(self.compare)
        self.assertEqual(inspected['signature']['arrival_options'], OPTIONS)
        self.assertEqual(inspected['signature']['selection_plan_sha256'],
                         sha(self.compare/'scripts/selection-plan.json'))
        legacy = preparation.prepare(self.root/'legacy-compare', self.legacy_build, '/unused', source_only=True)
        info = read(legacy/'submission.json'); info.update(executable=True, source_only=False)
        with mock.patch.object(reporter, 'verify_inputs', return_value=info):
            inspected = arrival_probe.inspect(legacy)
        self.assertNotIn('arrival_options', inspected['signature'])
        self.assertNotIn('selection_plan_sha256', inspected['signature'])

    def test_full_report_carries_exploration_and_conditional_bound(self):
        root = preparation.prepare(self.root/'synthetic', self.build, '/unused', source_only=True)
        with mock.patch.object(report_fixtures, 'routing', routing):
            info = report_fixtures.synthetic_evidence(root)
        with mock.patch.object(reporter, 'verify_inputs', return_value=info):
            gate = reporter.preflight(root, offline=True)
            result = reporter.report(root, offline=True)
        self.assertEqual(gate['arrival_options'], OPTIONS)
        self.assertEqual(result['arrival_options'], OPTIONS)
        self.assertEqual(result['timing_windows'], 36)
        self.assertEqual(result['formal_optimizer_steps'], 1800)
        for metadata in result['arrival_schedule'].values():
            self.assertEqual(metadata['conditional_error_multiplier'], 4.0)
            self.assertEqual(metadata['accepted_windows_with_positive_conservative_bound'], 0)
        text = (root/'report.md').read_text()
        self.assertIn('receiver-tail-assignment-v3', text)
        self.assertIn('4ε', text)
        self.assertIn('探索模式：True', text)


if __name__ == '__main__':
    unittest.main()
