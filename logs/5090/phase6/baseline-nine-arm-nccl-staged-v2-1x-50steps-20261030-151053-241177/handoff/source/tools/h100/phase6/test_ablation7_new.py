"""CPU regressions for the separate joint-assignment experiment and evidence path."""
import ast
import contextlib
import copy
import csv
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import ablation4_build
import ablation7_protocol as old_routing
import ablation7_new as cli
import ablation7_new_build as builder
import ablation7_new_plan as planner
import ablation7_new_prepare as preparation
import ablation7_new_protocol as routing
import ablation7_new_report as reporter
import ablation7_new_sources as sources
import arrival_assignment
import arrival_probe
import scope_plan
from common import read, sha, write
from test_protocol import mapping, reports
import test_ablation7_report as old_report_fixtures


class NewExperimentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ablation7-new-CPU-')
        cls.root = Path(cls.temp.name)
        cls.parent = ablation4_build.prepare(cls.root/'parent', source_only=True)
        cls.parent_hash = sha(cls.parent/'ablation4-manifest.json')
        cls.build = builder.prepare(cls.root/'build', cls.parent, source_only=True)
        cls.compare = preparation.prepare(cls.root/'compare', cls.build, '/unused', source_only=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_new_protocol_fits_assignment_and_leaves_old_protocol_unchanged(self):
        self.assertNotEqual(routing.PROTOCOL, old_routing.PROTOCOL)
        self.assertNotEqual(routing.B, old_routing.B)
        self.assertEqual(routing.definition(routing.B)['arrival_method'], 'tail_assignment')
        self.assertEqual(old_routing.definition(old_routing.B)['arrival_method'], 'tail_residual')
        plans = read(self.build/'plans.json')
        for role, order in plans['orders'].items():
            diagnostics = order['entry']['arrival_order']
            self.assertEqual(diagnostics['method'], arrival_assignment.METHOD)
            self.assertTrue(diagnostics['identity_fallback'])
            self.assertEqual(diagnostics['accepted_windows'], 0)
            original = plans['parent']['roles'][role]['entry']
            c = routing.role_plans(plans, routing.C)[role]['entry']
            self.assertEqual(c['coords'], original['coords'])
            for policy in (routing.A, routing.B):
                self.assertEqual(routing.role_plans(plans, policy)[role]['entry']['selected_per_source'], [448]*8)
        with self.assertRaises(ValueError):
            old_routing.role_plans(plans, old_routing.B)

    def test_identity_build_is_separate_but_compute_sources_match_a(self):
        record = builder.verify(self.build)
        self.assertEqual(sha(self.parent/'ablation4-manifest.json'), self.parent_hash)
        self.assertEqual({Path(p).name for p in record['inputs']}, set(builder.ALGORITHM_INPUTS))
        self.assertIn('arrival_assignment.py', builder.ALGORITHM_INPUTS)
        a = builder.runtime_root(self.build, routing.A)
        b = builder.runtime_root(self.build, routing.B)
        self.assertEqual(b.name, 'assignment_arrival_all')
        self.assertNotEqual(a, b)
        self.assertFalse((b/'libflux_cuda.so').exists())
        fused = read(self.build/'parent-fused-manifest.json')
        a_name = ablation4_build.FUSED_NAMES[builder.PARENT_POLICIES[routing.A]]
        self.assertEqual(record['improved_source_sha256'], fused['variants'][a_name]['source_sha256'])
        for policy in (routing.C, routing.A):
            self.assertEqual(builder.runtime_root(self.build, policy),
                             ablation4_build.runtime_root(self.parent, builder.PARENT_POLICIES[policy]))
        with self.assertRaisesRegex(ValueError, 'Unbuilt|diagnostic'):
            builder.verify(self.build, built=True)

    def test_prepare_dry_run_keeps_36_windows_even_with_identity_fallback(self):
        info = reporter.verify_inputs(self.compare)
        self.assertEqual(info['execution_timing'], dict(warmup_steps=10, timed_steps=50))
        self.assertEqual((info['blocks_per_round'], info['rounds'], info['timing_windows']), (6, 2, 36))
        self.assertEqual(set(info['variants']), set(routing.POLICIES))
        self.assertEqual(info['budget_fractions'], dict(zip(routing.POLICIES, (1/16, 1, 1))))
        with mock.patch.object(subprocess, 'run', side_effect=AssertionError('GPU invocation')), \
                contextlib.redirect_stdout(io.StringIO()):
            jobs = cli.run(self.compare)
            self.assertEqual(len(jobs), 42)
            self.assertEqual(sum(job['mode'] == 'timing' for job in jobs), 36)
            self.assertEqual(len(list(cli.mapping_jobs(self.compare, info))), 6)
            with self.assertRaisesRegex(ValueError, 'Source-only'):
                cli.run(self.compare, execute=True)
        self.assertFalse((self.compare/'all-state.json').exists())

    def test_generated_worker_has_new_gate_and_frozen_timing(self):
        for label, source in (('adapter', sources.adapter_source()), ('taco', sources.taco_source()),
                              ('worker', sources.worker_source()), ('fixture', sources.worker_source(True))):
            ast.parse(source, filename=label)
        source = sources.worker_source()
        with tempfile.TemporaryDirectory(prefix='ablation7-new-worker-') as temporary:
            path = Path(temporary)/'worker.py'; path.write_text(source)
            env = dict(os.environ, H100_ABLATION7_EXECUTE='1', H100_ABLATION4_EXECUTE='1')
            env.pop('H100_ABLATION7_NEW_EXECUTE', None)
            attempt = subprocess.run([sys.executable, str(path)], env=env, capture_output=True, text=True)
            self.assertNotEqual(attempt.returncode, 0)
            self.assertIn('GPU worker requires ablation7_new --execute', attempt.stderr)
            path.write_text(sources.worker_source(True))
            attempt = subprocess.run([sys.executable, str(path)],
                env=dict(env, H100_ABLATION7_NEW_EXECUTE='1'), capture_output=True, text=True)
            self.assertIn('Source-only ablation7_new worker', attempt.stderr)
        check = compile(source[source.index('from routing import PROTOCOL, execution_timing'):
                               source.index('CASE = next(')], '<new-worker-timing>', 'exec')
        config = dict(protocol=routing.PROTOCOL, execution_timing=routing.execution_timing(),
                      **routing.execution_timing())
        with mock.patch.dict(sys.modules, {'routing': routing}):
            exec(check, {'PLAN': config})
            for changed in (dict(config, protocol=old_routing.PROTOCOL), dict(config, timed_steps=60)):
                with self.assertRaises(ValueError): exec(check, {'PLAN': changed})

    def test_method_and_acceptance_diagnostics_cannot_be_relabelled(self):
        plans = read(self.build/'plans.json')
        for field, value in (('method', 'receiver-tail-residual-v1'), ('accepted_windows', 1),
                             ('identity_fallback', False), ('max_displacement', 9)):
            changed = copy.deepcopy(plans)
            order = changed['orders']['attn']
            order['entry']['arrival_order'][field] = value
            order['entry_sha256'] = routing.digest(order['entry'])
            with self.subTest(field=field), self.assertRaises(ValueError):
                planner.validate(changed)

    def test_new_report_exposes_assignment_fallback_without_swap_fields(self):
        plans = read(self.build/'plans.json')
        for role, metadata in reporter.arrival_metadata(plans).items():
            self.assertEqual(metadata['method'], arrival_assignment.METHOD)
            self.assertTrue(metadata['identity_fallback'])
            self.assertTrue(metadata['identical_to_A'])
            self.assertEqual(metadata['accepted_windows'], 0)
            self.assertEqual(metadata['changed_per_rank'], [0]*8)
            self.assertNotIn('accepted_swaps', metadata)
            self.assertTrue(metadata['window_status_counts'])

    def test_probe_routes_new_protocol_to_assignment_build(self):
        self.assertEqual(arrival_probe.REPORTERS[routing.PROTOCOL], 'ablation7_new_report')
        info = read(self.compare/'submission.json')
        info.update(executable=True, source_only=False)
        with mock.patch.object(reporter, 'verify_inputs', return_value=info):
            # Identity timing remains runnable, while the deployment gate must
            # recognize the new build and reject a claim of changed ordering.
            record = arrival_probe.inspect(self.compare)
            self.assertIs(record['reporter'], reporter)
            self.assertTrue(record['pair']['identity'])
            result = arrival_probe.probe([self.compare])
            self.assertEqual(result['status'], 'not_tested_identity')
            self.assertFalse(result['evaluated'])

    def test_full_synthetic_report_exercises_new_policy_and_metadata(self):
        root = preparation.prepare(self.root/'synthetic', self.build, '/unused', source_only=True)
        # Mock only fixture/admission in this CPU test; production code stays explicit.
        with mock.patch.object(old_report_fixtures, 'routing', routing):
            info = old_report_fixtures.synthetic_evidence(root)
        with mock.patch.object(reporter, 'verify_inputs', return_value=info):
            reporter.preflight(root, offline=True)
            result = reporter.report(root, offline=True)
        self.assertEqual(result['protocol'], routing.PROTOCOL)
        self.assertEqual(result['timing_windows'], 36)
        self.assertEqual(result['formal_optimizer_steps'], 1800)
        self.assertEqual(result['rounds']['combined']['blocks'], 12)
        self.assertIn(routing.B+'/'+routing.A, result['rounds']['combined']['comparisons'])
        self.assertIn('receiver-tail-assignment-v2', (root/'report.md').read_text())
        self.assertEqual(result['arrival_schedule']['attn']['accepted_windows'], 0)


class MeasuredReplayTests(unittest.TestCase):
    def test_hash_registered_raw_evidence_fits_joint_candidate_not_old_identity(self):
        hardware = scope_plan.frozen_mlp()['hardware']
        with tempfile.TemporaryDirectory(prefix='ablation7-new-raw-CPU-') as temporary:
            parent = dict(roles={})
            for role, expected_shape in routing.SHAPES.items():
                source = Path(temporary)/role; source.mkdir()
                record = scope_plan.cpu_fixture(role); config = record['config']
                rows, observations = mapping(config, 'interleaved'), reports(config)
                for rank, observation in enumerate(observations):
                    observation.update(protocol=scope_plan.PROTOCOL, run_id='CPU-synthetic', world_size=8,
                        policy='interleaved', sampling=True, smoke=False,
                        hardware=dict(hardware, local_rank=rank))
                    shape = observation['shapes'][0]; shape['K_global'] = expected_shape[2]
                    for sample in shape['samples']:
                        if sample['mode'] == 'receiver_sparse':
                            value = 1 if sample['offset'] < 2 else 11 if sample['offset'] in (2, 3) else 9
                            for fragment in sample['receiver']:
                                fragment['observed_ns'] = 1000000+value*1000+fragment['fragment']
                    write(source/f'b0-interleaved-instrumented-rank{rank}.json', observation)
                with (source/f'mapping-{role}.csv').open('w') as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
                write(source/'calibration-complete.json', dict(protocol=scope_plan.PROTOCOL,
                    completed=True, gpu_calibrated=True, run_id='CPU-synthetic', roles=[role], hardware=hardware))
                record.update(origin=dict(kind='scope_bf16_calibration', source=str(source), run_id='CPU-synthetic'),
                    hardware=hardware, inputs={str(p): sha(p) for p in source.iterdir()})
                record['entry'] = planner.fitter.fit(observations, rows, config, 'interleaved', expected_shape)
                old = planner.fitter.fit(observations, rows, config, 'interleaved', expected_shape,
                                         arrival_order='tail_residual')
                self.assertTrue(old['arrival_order']['identity_fallback'])
                parent['roles'][role] = record
            orders = planner.measured_orders(parent)
            for role, order in orders.items():
                self.assertEqual(order['entry']['arrival_order']['method'], arrival_assignment.METHOD)
                self.assertFalse(order['entry']['arrival_order']['identity_fallback'])
                self.assertGreater(order['entry']['arrival_order']['accepted_windows'], 0)
                self.assertEqual(order['entry']['mask'], parent['roles'][role]['entry']['mask'])
            evidence = Path(parent['roles']['attn']['origin']['source'])/'b0-interleaved-instrumented-rank0.json'
            evidence.write_text(evidence.read_text()+'\n')
            with self.assertRaises(ValueError): planner.measured_orders(parent)


if __name__ == '__main__':
    unittest.main()
