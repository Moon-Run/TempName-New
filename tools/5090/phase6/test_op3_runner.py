"""CPU protocol and supervisor checks; these fixtures are not GPU measurements."""
from collections import Counter
from contextlib import ExitStack, redirect_stdout
import copy
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import ModuleType
import unittest
from unittest.mock import patch

from rtx5090_loader import HERE
import op3 as controller
import op3_protocol as p
import op2_abc_build as optimizers


def sample(job):
    return dict(job, ms_per_step=900 if job['arm'] == p.ARMS[1] else 950,
                peak_allocated_gib=20, peak_reserved_gib=22,
                native_workspace_gib=0, peak_allocated_plus_taco_workspace_gib=20)


class Protocol(unittest.TestCase):
    def test_two_blocks_are_ab_ba_and_each_trial_is_independently_balanced(self):
        self.assertEqual(p.ARMS, ('op3-r', 'op3-improved'))
        self.assertEqual([j['arm'] for j in p.schedule()], [*p.ARMS, *p.ARMS[::-1]])
        self.assertEqual([j['mode'] for j in p.jobs()], ['smoke']*2+['profile']*2+['timing']*4)
        for count in (2, 4, 6, 8):
            jobs = p.schedule(trials=2, blocks=count)
            for trial in (1, 2):
                own = [j for j in jobs if j['trial'] == trial]
                self.assertEqual({j['block'] for j in own}, set(range(count)))
                for slot in (0, 1):
                    self.assertEqual(Counter(j['arm'] for j in own if j['index'] % 2 == slot),
                                     Counter(dict.fromkeys(p.ARMS, count//2)))
        for count in (0, 1, 3, 5, True, 2.0):
            with self.assertRaises(ValueError):
                p.schedule(blocks=count)

    def test_report_direction_and_ci_require_six_pairs_per_trial(self):
        with patch.object(p, 'definition', side_effect=lambda arm: {'arm': arm}):
            for blocks in (2, 4, 6):
                rows = [sample(j) for j in p.schedule(trials=2, blocks=blocks)]
                result = p.report(rows, trials=2, blocks=blocks)
                self.assertTrue(result['counterbalanced_order'])
                self.assertEqual(result['timing_windows'], blocks*4)
                self.assertEqual(result['formal_optimizer_steps'], blocks*200)
                self.assertFalse(result['increment_over_unoptimized_validated'])
                self.assertFalse(result['reduction_changes_allowed'])
                self.assertTrue(result['raw_projection_bitwise_required'])
                self.assertTrue(result['smoke_training_equivalence_required'])
                self.assertFalse(result['full_training_equivalence_required'])
                for comparison in result['comparisons'].values():
                    pairs = comparison['pairs']
                    self.assertEqual(list(pairs), ['op3-improved/op3-r'])
                    value = pairs['op3-improved/op3-r']
                    self.assertAlmostEqual(value['percent'], 100*(1-900/950))
                    self.assertEqual(value['faster_blocks'], blocks)
                    self.assertEqual(value['ci95'] is None, blocks < 6)
                    self.assertEqual(value['paired_blocks'], blocks)
                self.assertIn('2 组', p.markdown(result))
                self.assertIn('AB/BA', p.markdown(result))
                with self.assertRaises(ValueError):
                    p.report(rows[:-1], trials=2, blocks=blocks)
                changed = copy.deepcopy(rows)
                changed[0], changed[1] = changed[1], changed[0]
                with self.assertRaises(ValueError):
                    p.report(changed, trials=2, blocks=blocks)

    def test_negative_gains_are_preserved_and_invalid_pairs_rejected(self):
        self.assertLess(p.paired_summary([101, 102], [100, 100])['percent'], 0)
        for candidate, reference in (([1], [1]), ([1, 2], [1]), ([1, 0], [1, 2]),
                                     ([1, float('nan')], [1, 2]), ([1]*3, [1]*3)):
            with self.assertRaises(ValueError):
                p.paired_summary(candidate, reference)

    def test_numerical_comparison_distinguishes_consumer_stage_and_keeps_training_evidence(self):
        row = dict(losses=[1.0, 0.99], grad_norms=[2.0, 1.9], final_parameters_sha256='a'*64,
                   parameter_trajectory=[{'parameter': 'b'*64}], trainable_parameters={'parameter': [1]},
                   smoke_output_sha256={'layer': 'c'*64})
        reference, candidate = copy.deepcopy(row), copy.deepcopy(row)
        candidate['smoke_output_sha256']['layer'] = 'd'*64
        references = {(0, -1, 'smoke'): {p.ARMS[0]: [reference]*8, p.ARMS[1]: [candidate]*8}}
        result = controller.numerical_comparison(references)['op3-improved/op3-r']
        self.assertTrue(result['all_trajectories_identical'])
        self.assertTrue(result['output_stages_differ'])
        self.assertFalse(result['reduction_changes_allowed'])
        self.assertTrue(result['raw_projection_bitwise_required'])
        self.assertFalse(result['full_training_equivalence_required'])
        self.assertIsNone(result['windows'][0]['smoke_projection_outputs_identical'])
        self.assertTrue(result['windows'][0]['smoke_parameter_trajectory_identical'])
        self.assertEqual(result['max_loss_abs_difference'], 0)
        self.assertEqual(candidate['smoke_output_sha256']['layer'], 'd'*64)
        import op3_validation as validation
        with patch.object(validation, 'identity', return_value='CPU-FIXTURE'):
            paired = {}
            for arm, rows in references[(0, -1, 'smoke')].items():
                controller.check_identity(rows, None, paired,
                    dict(trial=0, block=-1, mode='smoke', arm=arm))
        self.assertEqual(set(paired[(0, -1, 'smoke')]), set(p.ARMS))


class Launcher(unittest.TestCase):
    def test_worker_environment_maps_both_policies_to_registered_native_build(self):
        builder = ModuleType('op3_build')
        builder.base_root = lambda root: Path('/CPU/baseline')
        builder.native_root = lambda root, arm: Path('/CPU/interleaved')
        parent_env = dict(PYTHONPATH='/CPU/basic:/CPU/scripts',
                          LD_LIBRARY_PATH='/CPU/basic:/CPU/cuda',
                          CUDA_DEVICE_MAX_CONNECTIONS='1', RTX5090_BASELINE_EXECUTE='1')
        with patch.dict(sys.modules, {'op3_build': builder}), \
                patch.object(controller.load('build'), 'native_root', return_value=Path('/CPU/basic')), \
                patch.object(controller.base, 'worker_environment', side_effect=lambda *a: dict(parent_env)) as base_env:
            for arm in p.ARMS:
                env = controller.worker_environment(None, Path('/CPU/op3'), None, {'arm': arm}, None, {})
                self.assertEqual(env['E2E_POLICY'], arm)
                self.assertEqual(env['RTX5090_OP3_EXECUTE'], '1')
                self.assertEqual(env['CUDA_DEVICE_MAX_CONNECTIONS'], '1')
                self.assertEqual(env['RTX5090_BASELINE_EXECUTE'], '1')
                self.assertEqual(env['PYTHONPATH'], '/CPU/interleaved:/CPU/scripts')
                self.assertEqual(env['LD_LIBRARY_PATH'], '/CPU/interleaved:/CPU/cuda')
                self.assertEqual(base_env.call_args.args[3]['arm'], 'flux-basic')

    def test_both_entrypoints_are_preview_only_and_create_no_files(self):
        for launcher in ('run_op3.sh', 'run_op3_compare.sh'):
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'absent-run'
                proc = subprocess.run(['bash', str(HERE/launcher), '--out', str(out),
                    '--abc-build', str(Path(tmp)/'absent-build')], text=True, capture_output=True,
                    env=dict(os.environ, RTX5090_PYTHON=sys.executable))
                self.assertEqual(proc.returncode, 0, proc.stderr)
                record = json.loads(proc.stdout[proc.stdout.index('{'):])
                self.assertEqual(record['arms'], list(p.ARMS))
                self.assertEqual(record['timing_windows'], 4)
                self.assertEqual(record['formal_optimizer_steps'], 200)
                self.assertTrue(record['counterbalanced_order'])
                self.assertFalse(out.exists())

    def test_execute_requires_explicit_parent_but_preflight_does_not(self):
        with self.assertRaisesRegex(ValueError, '--abc-build or --op3-build'):
            controller.arguments(['--execute'])
        self.assertTrue(controller.arguments(['--execute', '--preflight-only']).preflight_only)
        self.assertEqual(controller.arguments([]).blocks, 2)
        with self.assertRaises(ValueError):
            controller.arguments(['--blocks', '3'])

    def test_early_stage_preview_reports_zero_timing_windows(self):
        for flag in ('--preflight-only', '--transport-smoke-only'):
            with redirect_stdout(io.StringIO()) as output:
                controller.main([flag])
            text = output.getvalue()
            record = json.loads(text[text.index('{'):])
            self.assertEqual(record['timing_windows'], 0)
            self.assertEqual(record['formal_optimizer_steps'], 0)


class Supervisor(unittest.TestCase):
    def execute_fixture(self, root, *, stop=None, fail=None, fresh=False, tamper=False,
                        real_sources=False, smoke_drift=None):
        """Run the actual controller loop with GPU-facing boundaries replaced."""
        build = root/'existing-build'
        build.mkdir()
        environment = {'torch': optimizers.compat.TORCH}
        controller.write(build/'op3-build.json', {'environment': environment})
        controller.write(root/'batch-state.json', {})
        args = controller.arguments(['--out', str(root), '--execute',
            '--abc-build' if fresh else '--op3-build', str(build), *([stop] if stop else [])])
        modules = {name: ModuleType(name) for name in
                   ('op3_build', 'op3_sources', 'op3_profile', 'run', 'tp8_dp2_train')}
        actual_sources = None
        if real_sources:
            import op3_sources as actual_sources
            modules['op3_sources'] = actual_sources
        calls, setup = [], []
        builder = modules['op3_build']
        builder.verify = lambda *a: None
        builder.base_root = lambda *a: root
        def prepare(path, parent):
            setup.append('prepare')
            path.mkdir()
            controller.write(path/'op3-build.json', {'environment': environment})
        builder.prepare = prepare
        builder.compile_build = lambda *a: setup.append('compile')
        if not real_sources:
            for name in ('worker_source', 'adapter_source', 'taco_source'):
                setattr(modules['op3_sources'], name, lambda arm: 'pass')
            modules['op3_sources'].prepare_workers = lambda *a: dict.fromkeys(p.ARMS, {})
            modules['op3_sources'].definition = lambda arm: {'arm': arm}
        def analyze(events, arm, rank):
            if fail == 'profile' and arm == p.ARMS[1]:
                raise ValueError('fixture profile failure')
            return dict(execution_validated=fail != 'profile-rejected',
                        arm='foreign' if fail == 'profile-foreign' else arm, rank=rank)
        modules['op3_profile'].analyze = analyze
        modules['run'].arguments = lambda *a: []
        def run(command, cwd, env, log, timeout, peer):
            dest = Path(log).parent
            job = controller.read(dest/'command.json')['job']
            calls.append(job)
            if job['mode'] == 'timing':
                stored = controller.read(root/'profile-summary.json')
                self.assertTrue(stored['completed'])
                self.assertTrue(stored['execution_validated'])
                self.assertEqual(len(stored['profiles']), 16)
            if job['mode'] == 'profile':
                for rank in range(8):
                    with gzip.open(dest/f'rank{rank}-profile.json.gz', 'wt') as handle:
                        json.dump({'traceEvents': []}, handle)
            if tamper and len(calls) == 1:
                controller.write(build/'op3-build.json', {'environment': environment, 'changed': True})
        modules['tp8_dp2_train'].run_process = run
        def hardware(args):
            setup.append('hardware')
            record = dict(environment=environment, communication={}, cuda_visible_devices='0,1,2,3,4,5,6,7')
            controller.write(root/'environment-preflight.json', record)
            return record
        def native_checks(*args):
            setup.append('native-probe')
            if fail == 'native':
                raise ValueError('fixture native failure')
        def window(root, job, *rest):
            if fail == 'smoke' and job['mode'] == 'smoke':
                raise ValueError('fixture smoke failure')
            if smoke_drift:
                row = dict(losses=[1.0], grad_norms=[2.0], final_parameters_sha256='a'*64,
                    parameter_trajectory=[{'gradient': 'b'*64, 'parameter': 'c'*64}],
                    trainable_parameters={'parameter': [1]})
                if job['mode'] == 'smoke' and job['arm'] == p.ARMS[1]:
                    if smoke_drift == 'grad_norms':
                        row['grad_norms'] = [2.01]
                    elif smoke_drift == 'final_parameters_sha256':
                        row['final_parameters_sha256'] = 'd'*64
                    else:
                        row['parameter_trajectory'][0]['gradient'] = 'e'*64
                return [copy.deepcopy(row) for _ in range(8)]
            return []
        def ma11_probe(*args):
            setup.append('ma11-probe')
            if fail == 'ma11':
                raise ValueError('fixture ma11 failure')
            return {'fixture': True}
        with ExitStack() as stack:
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(controller, 'REPO', root))
            for name in ('verify_controllers', 'gpu_idle', 'verify_ma11_probe'):
                stack.enter_context(patch.object(controller.base, name))
            stack.enter_context(patch.object(controller.base, 'check_worker', side_effect=lambda root, arm, worker: root/'workers'/arm/'scripts'))
            stack.enter_context(patch.object(controller.base, 'hardware_preflight', side_effect=hardware))
            stack.enter_context(patch.object(controller.load('build'), 'environment_identity', return_value=environment))
            from test_op3_sources import library_plan
            records = library_plan(root)
            stack.enter_context(patch.object(optimizers, 'prepare', side_effect=lambda *a: setup.append('ma9') or records['ma9_compat']))
            stack.enter_context(patch.object(optimizers.compat, 'prepare', side_effect=lambda *a: setup.append('ma11') or records['ma11_compat']))
            if real_sources:
                stack.enter_context(patch.object(actual_sources.builder, 'codec_library',
                    return_value={'path': '/CPU/libop3_codec.so', 'sha256': 'c'*64}))
                stack.enter_context(patch.object(actual_sources.builder, 'native_root', return_value=build))
                stack.enter_context(patch.object(actual_sources.builder, 'libraries', return_value={}))
            stack.enter_context(patch.object(controller.base, 'ma11_gpu_preflight', side_effect=ma11_probe))
            stack.enter_context(patch.object(optimizers, 'verify_all'))
            stack.enter_context(patch.object(controller, 'native_checks', side_effect=native_checks))
            stack.enter_context(patch.object(controller, 'worker_environment', return_value={}))
            stack.enter_context(patch.object(controller, 'window', side_effect=window))
            if smoke_drift:
                import op3_validation as validation
                stack.enter_context(patch.object(validation, 'identity', return_value='CPU-FIXTURE'))
            else:
                stack.enter_context(patch.object(controller, 'check_identity', return_value='CPU-FIXTURE'))
            stack.enter_context(patch.object(p, 'summarize', side_effect=lambda rows, job: sample(job)))
            result = p.report([sample(j) for j in p.schedule()])
            stack.enter_context(patch.object(controller, 'replay', return_value=result))
            try:
                with redirect_stdout(io.StringIO()):
                    controller.supervise(args)
            except ValueError as exc:
                return calls, setup, exc
        return calls, setup, None

    def test_complete_lifecycle_runs_all_gates_then_two_smokes_profiles_and_four_windows(self):
        for fresh in (False, True):
            with self.subTest(fresh=fresh), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, setup, error = self.execute_fixture(root, fresh=fresh)
                self.assertIsNone(error)
                self.assertEqual(setup, ['hardware']+(['prepare', 'compile'] if fresh else [])+
                                 ['ma9', 'ma11', 'ma11-probe', 'native-probe'])
                self.assertEqual([j['mode'] for j in calls], ['smoke']*2+['profile']*2+['timing']*4)
                self.assertEqual({j['arm'] for j in calls}, set(p.ARMS))
                self.assertEqual(controller.read(root/'batch-state.json')['status'], 'completed')
                self.assertEqual(controller.read(root/'report.json')['formal_optimizer_steps'], 200)
                self.assertEqual(len(controller.read(root/'profile-summary.json')['profiles']), 16)

    def test_failed_gates_never_admit_timing(self):
        for failure, expected_modes in (
                ('ma11', []), ('native', []), ('smoke', ['smoke']),
                ('profile', ['smoke']*2+['profile']*2),
                ('profile-rejected', ['smoke']*2+['profile']),
                ('profile-foreign', ['smoke']*2+['profile'])):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, setup, error = self.execute_fixture(root, fail=failure)
                self.assertIsNotNone(error)
                self.assertEqual([j['mode'] for j in calls], expected_modes)
                self.assertEqual(controller.read(root/'batch-state.json')['status'], 'failed')
                self.assertFalse((root/'report.json').exists())

    def test_controller_integrates_actual_generated_packages_and_registered_definitions(self):
        import op3_sources as sources
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, setup, error = self.execute_fixture(root, real_sources=True)
            self.assertIsNone(error)
            self.assertEqual([j['mode'] for j in calls], ['smoke']*2+['profile']*2+['timing']*4)
            workers = controller.read(root/'workers.json')
            self.assertEqual(set(workers), set(p.ARMS))
            for arm, worker in workers.items():
                self.assertEqual(worker['config']['op3_implementation'], sources.implementation(arm))
                self.assertEqual(worker['config']['op3_optimization_stack'], sources.STACK)
                self.assertEqual(set(worker['config']['op3_optimization_libraries']), {'ma9', 'ma11'})
                self.assertEqual((Path(worker['root'])/'worker.py').read_text(), sources.worker_source(arm))
            result = controller.read(root/'report.json')
            self.assertEqual(result['definitions'], {arm: sources.definition(arm) for arm in p.ARMS})

    def test_smoke_gradient_or_parameter_drift_blocks_profiles_and_timing(self):
        for field in ('grad_norms', 'parameter_trajectory', 'final_parameters_sha256'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, setup, error = self.execute_fixture(root, smoke_drift=field)
                self.assertIn('changed loss, gradients or parameter updates', str(error))
                self.assertEqual([job['mode'] for job in calls], ['smoke', 'smoke'])
                self.assertEqual(controller.read(root/'batch-state.json')['status'], 'failed')
                self.assertFalse((root/'profile-summary.json').exists())
                self.assertFalse((root/'report.json').exists())

    def test_modified_build_is_rejected_before_next_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, setup, error = self.execute_fixture(root, tamper=True)
            self.assertIn('build identity changed', str(error))
            self.assertEqual(len(calls), 1)
            self.assertEqual(controller.read(root/'batch-state.json')['status'], 'failed')

    def test_early_stops_do_not_claim_completed_benchmark(self):
        for flag, expected in (('--preflight-only', 'preflight_completed'),
                               ('--transport-smoke-only', 'transport_smoke_completed')):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, setup, error = self.execute_fixture(root, stop=flag)
                self.assertIsNone(error)
                self.assertEqual(calls, [])
                state = controller.read(root/'batch-state.json')
                self.assertEqual(state['status'], expected)
                self.assertFalse(state['gpu_benchmark_completed'])
                self.assertFalse((root/'report.json').exists())
                if flag == '--preflight-only':
                    self.assertEqual(setup, ['hardware'])


if __name__ == '__main__':
    unittest.main()
