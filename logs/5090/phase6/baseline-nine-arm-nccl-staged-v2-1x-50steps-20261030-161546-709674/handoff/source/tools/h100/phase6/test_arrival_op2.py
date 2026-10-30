"""CPU checks for identical codec attribution and sealed scheduling-only evidence."""
import ast
import contextlib
import copy
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ablation3_plan
import ablation4_build
import arrival_pair_order as order
import arrival_op2 as cli
import arrival_op2_build as builder
import arrival_op2_plan as planner
import arrival_op2_prepare as preparation
import arrival_op2_protocol as routing
import arrival_op2_report as reporter
import arrival_op2_sources as sources
import scope_plan
import test_ablation7_report as report_fixtures
from common import read, sha, write


class PlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parent = scope_plan.bundle(source_only=True)
        cls.plans = planner.bundle(cls.parent, source_only=True, spans={'attn': 4, 'mlp': 8})

    def test_original_a_and_every_quantization_input_are_preserved(self):
        expected = ablation3_plan.derive(self.parent, routing.A)['roles']
        self.assertEqual(routing.role_plans(self.plans, routing.A), expected)
        self.assertNotIn('scores', self.plans)
        self.assertNotIn('arrival_options', self.plans)
        self.assertEqual(routing.PRIMARY, [(routing.S, routing.A)])
        for role, a in expected.items():
            s = routing.role_plans(self.plans, routing.S)[role]
            self.assertEqual(a['mask_sha256'], s['mask_sha256'])
            self.assertNotEqual(a['entry']['coords'], s['entry']['coords'])
            for key in set(a['entry']) - {'coords', 'maps'}:
                self.assertEqual(a['entry'][key], s['entry'][key])
            self.assertEqual(s['entry']['selected_per_source'], [448]*8)

    def test_all_parameter_combinations_preserve_windows_and_identity_fallback(self):
        for attn in order.SPANS:
            for mlp in order.SPANS:
                plans = planner.bundle(self.parent, source_only=True, spans=dict(attn=attn, mlp=mlp))
                planner.validate(plans)
        identity = planner.bundle(self.parent, source_only=True)
        self.assertEqual(routing.role_plans(identity, routing.A), routing.role_plans(identity, routing.S))
        self.assertTrue(all(d['identity_fallback'] for d in identity['schedule_diagnostics'].values()))

    def test_span_and_quantization_drift_are_rejected(self):
        for options in (None, {'attn': 2, 'mlp': 4}, {'attn': True, 'mlp': 4}, {'attn': 4}):
            bad = dict(self.plans, schedule_options=options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                planner.validate(bad)
        bad = copy.deepcopy(self.plans)
        role = routing.role_plans(bad, routing.S)['mlp']
        role['entry']['mask'][0][64] = 0
        role['entry_sha256'] = routing.digest(role['entry'])
        with self.assertRaises(ValueError):
            planner.validate(bad)
        bad = copy.deepcopy(self.plans)
        bad['schedule_diagnostics']['attn']['changed_per_rank'] = [0]*8
        with self.assertRaises(ValueError):
            planner.validate(bad)

    def test_same_codec_decoder_frontend_barriers_in_both_arms(self):
        a, s = (routing.definition(p) for p in routing.POLICIES)
        changes = {key for key in a if a[key] != s[key]}
        self.assertEqual(changes, {'policy', 'family', 'order', 'scheduling_method'})
        for policy in routing.POLICIES:
            d = routing.definition(policy)
            self.assertTrue(routing.is_fused_v2(policy))
            self.assertFalse(routing.is_native(policy))
            self.assertEqual(d['implementation'], 'a800-fused-v2-cta-v1')
            self.assertEqual(d['decoder'], 'cta128')
            self.assertEqual(d['workspace'], 'single')
            self.assertTrue(d['output_clone'] and d['publication_barrier'] and d['reuse_barrier'])

    def test_balanced_two_arm_schedule_and_round_reversal(self):
        jobs = routing.schedule()
        self.assertEqual(len(jobs), 24)
        first, second = jobs[:12], jobs[12:]
        self.assertEqual([j['policy'] for j in second], [j['policy'] for j in first][::-1])
        for block in range(6):
            self.assertEqual({j['policy'] for j in first if j['block'] == block}, set(routing.POLICIES))
        for position in range(2):
            self.assertEqual({p: sum(j['policy'] == p and j['index'] % 2 == position for j in first)
                              for p in routing.POLICIES}, {p: 3 for p in routing.POLICIES})

    def test_acceptance_requires_point_ci_and_changed_schedule(self):
        key = routing.S+'/'+routing.A
        def check(point, lower, upper=2.0, changed=True):
            return reporter.acceptance(dict(comparisons={key: dict(percent=point, ci95=[lower, upper])}), changed)
        self.assertFalse(check(1.6, 1.49)['passed'])
        self.assertFalse(check(1.49, 1.5)['passed'])
        self.assertFalse(check(1.6, 1.5, changed=False)['passed'])
        self.assertTrue(check(1.5, 1.5)['passed'])
        self.assertTrue(check(2.0, 1.7, 2.3)['passed'])
        self.assertFalse(reporter.acceptance(dict(comparisons={key: dict(percent=2, ci95=None)}))['passed'])

    def test_acceptance_rejects_invalid_statistics(self):
        key = routing.S+'/'+routing.A
        for value in (True, '2', float('nan'), float('inf')):
            with self.subTest(point=value), self.assertRaises(ValueError):
                reporter.acceptance(dict(comparisons={key: dict(percent=value, ci95=[1.6, 2.1])}))
        for interval in ([2, 1], [1], [1, 2, 3], [True, 2], ['1', 2],
                         [1, float('nan')], [1, float('inf')], '1,2'):
            with self.subTest(interval=interval), self.assertRaises(ValueError):
                reporter.acceptance(dict(comparisons={key: dict(percent=2, ci95=interval)}))

    def test_generated_worker_gates_execution_and_schedule_environment(self):
        for source in (sources.adapter_source(), sources.taco_source(),
                       sources.worker_source(), sources.worker_source(True)):
            ast.parse(source)
        self.assertIn('Runtime scheduling environment differs from the frozen selection', sources.worker_source())
        with tempfile.TemporaryDirectory(prefix='op2-worker-CPU-') as temporary:
            worker = Path(temporary)/'worker.py'
            worker.write_text(sources.worker_source())
            env = dict(os.environ, H100_ABLATION4_EXECUTE='1', H100_ABLATION7_NEW_EXECUTE='1')
            env.pop('H100_ARRIVAL_OP2_EXECUTE', None)
            result = subprocess.run([sys.executable, str(worker)], env=env, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('GPU worker requires arrival_op2 --execute', result.stderr)
            worker.write_text(sources.worker_source(True))
            result = subprocess.run([sys.executable, str(worker)],
                                    env=dict(env, H100_ARRIVAL_OP2_EXECUTE='1'),
                                    capture_output=True, text=True)
            self.assertIn('Source-only arrival_op2 worker', result.stderr)


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='arrival-op2-CPU-')
        cls.root = Path(cls.temp.name)
        cls.parent = ablation4_build.prepare(cls.root/'parent', source_only=True)
        cls.parent_hash = sha(cls.parent/'ablation4-manifest.json')
        cls.build = builder.prepare(cls.root/'build', cls.parent, source_only=True)
        cls.compare = preparation.prepare(cls.root/'compare', cls.build, '/unused', source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_only_scheduler_sources_differ_and_template_is_identity(self):
        record = builder.verify(self.build)
        self.assertEqual(sha(self.parent/'ablation4-manifest.json'), self.parent_hash)
        fused = read(self.build/'parent-fused-manifest.json')
        base = fused['variants']['interleaved_all']['source_sha256']
        s = record['improved_source_sha256']['paired_order']
        self.assertEqual({name for name in set(base) | set(s) if base.get(name) != s.get(name)},
                         set(order.ALLOWED_SOURCE_CHANGES))
        self.assertEqual(base[builder.RUNTIME], s[builder.RUNTIME])
        self.assertEqual(builder.runtime_root(self.build, routing.A), self.parent/'fused/interleaved_all')
        self.assertFalse((builder.runtime_root(self.build, routing.S)/'libflux_cuda.so').exists())
        self.assertEqual(read(self.build/'plans.json')['schedule_options'], {'attn': 32, 'mlp': 32})
        self.assertEqual(read(self.compare/'scripts/selection-plan.json')['schedule_options'], {'attn': 4, 'mlp': 8})
        with self.assertRaises(ValueError):
            builder.verify(self.build, built=True)

    def test_prepare_and_dry_run_cannot_execute_cpu_fixture(self):
        info = reporter.verify_inputs(self.compare)
        self.assertEqual(info['timing_windows'], 24)
        self.assertEqual(info['budget_fractions'], {p: 1 for p in routing.POLICIES})
        self.assertFalse(read(self.compare/'scripts/config.json')['double_buffered'])
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('GPU invocation')), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(len(cli.run(self.compare)), 28)
            self.assertEqual(len(list(cli.mapping_jobs(self.compare, info))), 4)
        for function in (reporter.report, reporter.preflight):
            with self.assertRaisesRegex(ValueError, 'CPU rehearsal'):
                function(self.compare, offline=True)

    def test_source_selection_and_schedule_drift_rejected(self):
        for relative in ('scripts/worker.py', 'provenance/selection/selection.json', 'provenance/build-plans.json'):
            path = self.compare/relative
            saved = path.read_bytes()
            try:
                path.write_bytes(saved+b'\n')
                with self.subTest(relative=relative), self.assertRaises(ValueError):
                    reporter.verify_inputs(self.compare, offline=True)
            finally:
                path.write_bytes(saved)
        path = self.compare/'submission.json'; saved = read(path)
        try:
            changed = copy.deepcopy(saved)
            changed['schedule'][0]['policy'] = changed['schedule'][1]['policy']
            write(path, changed)
            with self.assertRaises(ValueError):
                reporter.verify_inputs(self.compare, offline=True)
        finally:
            write(path, saved)

    def test_offline_uses_frozen_build_and_probe_snapshots(self):
        hidden = self.root/'build-hidden'; parent_hidden = self.root/'parent-hidden'
        self.build.rename(hidden); self.parent.rename(parent_hidden)
        try:
            reporter.verify_inputs(self.compare, offline=True)
            with self.assertRaises((ValueError, FileNotFoundError)):
                reporter.verify_inputs(self.compare)
        finally:
            hidden.rename(self.build); parent_hidden.rename(self.parent)

    def test_profile_rejects_the_abandoned_warp_decoder(self):
        for policy in routing.POLICIES:
            with mock.patch.object(report_fixtures.fixtures, 'sp', routing):
                events = report_fixtures.fixtures.profile_fixture(policy)
            reporter.profile_roles(events, policy)
            for event in events:
                if event.get('name') == 'taco_decode_ring_kernel':
                    event['name'] = 'taco_decode_ring_kernel_op2'
            with self.assertRaisesRegex(ValueError, 'unchanged original CTA decoder'):
                reporter.profile_roles(events, policy)

    def test_complete_synthetic_report_attributes_only_scheduling(self):
        root = preparation.prepare(self.root/'synthetic', self.build, '/unused', source_only=True)
        with mock.patch.object(report_fixtures, 'routing', routing):
            info = report_fixtures.synthetic_evidence(root)
        # Mock only GPU admission. Every timing/profile/mapping record is still
        # parsed and all source-only workers remain permanently non-executable.
        for path in (root/'results').glob('**/rank*.json'):
            row = read(path)
            row['schedule_options'] = info['schedule_options']
            row['runtime_spans'] = routing.IDENTITY_SPANS if row['policy'] == routing.A else info['schedule_options']
            write(path, row)
        with mock.patch.object(reporter, 'verify_inputs', return_value=info):
            gate = reporter.preflight(root, offline=True)
            result = reporter.report(root, offline=True)
        self.assertNotIn('decoder-validation.json', gate['files'])
        self.assertEqual(result['protocol'], routing.PROTOCOL)
        self.assertEqual(result['timing_windows'], 24)
        self.assertEqual(result['formal_optimizer_steps'], 1200)
        self.assertEqual(result['timing_rank_records'], 192)
        self.assertEqual(result['rounds']['combined']['blocks'], 12)
        self.assertEqual(result['acceptance_threshold'], 1.5)
        self.assertEqual(result['acceptance']['candidate'], routing.S)
        self.assertFalse(result['arrival_diagnostic_only'])
        self.assertEqual(set(result['acceptance_by_round']), {'1', '2', 'combined'})
        self.assertIn('相同的编码器', (root/'report.md').read_text())
        self.assertFalse(result['arrival_schedule']['attn']['identical_to_A'])
        self.assertEqual(result['schedule_options'], info['schedule_options'])
        self.assertEqual(result['selection_sha256'], info['selection']['sha256'])


if __name__ == '__main__':
    unittest.main()
