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
    return dict(job, ms_per_step=950,
                peak_allocated_gib=20, peak_reserved_gib=22,
                native_workspace_gib=0, peak_allocated_plus_taco_workspace_gib=20)


class Protocol(unittest.TestCase):
    def test_only_combined_runs_and_positive_odd_blocks_are_valid(self):
        self.assertEqual(p.ARMS, ('op3-combined',))
        self.assertEqual([j['mode'] for j in p.jobs()], ['smoke', 'profile', 'timing'])
        for count in (1, 2, 3, 20):
            jobs = p.schedule(trials=2, blocks=count)
            self.assertEqual(len(jobs), count*2)
            self.assertEqual({j['arm'] for j in jobs}, {p.COMBINED})
            for trial in (1, 2):
                self.assertEqual([j['block'] for j in jobs if j['trial'] == trial], list(range(count)))
        for count in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                p.schedule(blocks=count)

    def test_single_arm_report_never_invents_paired_or_historical_gain(self):
        for blocks in (1, 2, 20):
            samples = [sample(job) for job in p.schedule(2, blocks)]
            result = p.report(samples, 2, blocks)
            self.assertEqual(result['timing_windows'], blocks*2)
            self.assertEqual(result['formal_optimizer_steps'], blocks*100)
            self.assertFalse(result['counterbalanced_order'])
            self.assertFalse(result['pairwise_ci_available'])
            self.assertFalse(result['increment_over_unoptimized_validated'])
            self.assertTrue(result['operator_gradient_equivalence_required'])
            for value in result['comparisons'].values():
                self.assertEqual(value['pairs'], {})
                self.assertEqual(set(value['arms']), {p.COMBINED})
            with self.assertRaises(ValueError):
                p.report(samples[:-1], 2, blocks)
            changed = copy.deepcopy(samples)
            changed[0]['arm'] = 'op3-r'
            with self.assertRaises(ValueError):
                p.report(changed, 2, blocks)
            self.assertIn('1 组', p.markdown(result))


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
        for launcher in ('run_op3.sh', 'run_op3_compare.sh', 'run_op3_operator.sh'):
            with self.subTest(launcher=launcher), tempfile.TemporaryDirectory() as tmp:
                out = Path(tmp)/'absent-run'
                proc = subprocess.run(['bash', str(HERE/launcher), '--out', str(out),
                    '--abc-build', str(Path(tmp)/'absent-build')], text=True, capture_output=True,
                    env=dict(os.environ, RTX5090_PYTHON=sys.executable))
                self.assertEqual(proc.returncode, 0, proc.stderr)
                record = json.loads(proc.stdout[proc.stdout.index('{'):])
                self.assertEqual(record['arms'], list(p.ARMS))
                self.assertEqual(record['timing_windows'], 0 if launcher == 'run_op3_operator.sh' else 1)
                self.assertEqual(record['formal_optimizer_steps'], 0 if launcher == 'run_op3_operator.sh' else 50)
                self.assertFalse(record['counterbalanced_order'])
                self.assertEqual(record['operator_settings']['windows'], 1)
                self.assertFalse(out.exists())

    def test_execute_requires_explicit_parent_but_preflight_does_not(self):
        with self.assertRaisesRegex(ValueError, '--abc-build or --op3-build'):
            controller.arguments(['--execute'])
        self.assertTrue(controller.arguments(['--execute', '--preflight-only']).preflight_only)
        self.assertEqual(controller.arguments([]).blocks, 1)
        self.assertEqual(controller.arguments(['--blocks', '3']).blocks, 3)

    def test_early_stage_preview_reports_zero_timing_windows(self):
        for flag in ('--preflight-only', '--transport-smoke-only', '--operator-only'):
            with redirect_stdout(io.StringIO()) as output:
                controller.main([flag])
            text = output.getvalue()
            record = json.loads(text[text.index('{'):])
            self.assertEqual(record['timing_windows'], 0)
            self.assertEqual(record['formal_optimizer_steps'], 0)


class Supervisor(unittest.TestCase):
    def execute_fixture(self, root, *, stop=None, fail=None, fresh=False, tamper=False,
                        real_sources=False):
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
            if fail == 'profile':
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
                self.assertEqual(len(stored['profiles']), 8)
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
            return []
        def operator_checks(*args):
            setup.append('operator')
            if fail == 'operator':
                raise ValueError('fixture operator gradient failure')
            result = dict(completed=True, operator_gradients_validated=fail != 'operator-rejected', arm=p.COMBINED)
            controller.write(root/'operator/report.json', result)
            return result
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
            stack.enter_context(patch.object(controller, 'operator_checks', side_effect=operator_checks))
            stack.enter_context(patch.object(controller, 'worker_environment', return_value={}))
            stack.enter_context(patch.object(controller, 'window', side_effect=window))
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

    def test_complete_lifecycle_runs_all_gates_then_single_training_window(self):
        for fresh in (False, True):
            with self.subTest(fresh=fresh), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, setup, error = self.execute_fixture(root, fresh=fresh)
                self.assertIsNone(error)
                self.assertEqual(setup, ['hardware']+(['prepare', 'compile'] if fresh else [])+
                                 ['ma9', 'ma11', 'ma11-probe', 'native-probe', 'operator'])
                self.assertEqual([j['mode'] for j in calls], ['smoke', 'profile', 'timing'])
                self.assertEqual({j['arm'] for j in calls}, set(p.ARMS))
                self.assertEqual(controller.read(root/'batch-state.json')['status'], 'completed')
                self.assertEqual(controller.read(root/'report.json')['formal_optimizer_steps'], 50)
                self.assertEqual(len(controller.read(root/'profile-summary.json')['profiles']), 8)

    def test_failed_gates_never_admit_timing(self):
        for failure, expected_modes in (
                ('ma11', []), ('native', []), ('operator', []), ('operator-rejected', []), ('smoke', ['smoke']),
                ('profile', ['smoke', 'profile']),
                ('profile-rejected', ['smoke', 'profile']),
                ('profile-foreign', ['smoke', 'profile'])):
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
            self.assertEqual([j['mode'] for j in calls], ['smoke', 'profile', 'timing'])
            workers = controller.read(root/'workers.json')
            self.assertEqual(set(workers), set(p.ARMS))
            for arm, worker in workers.items():
                self.assertEqual(worker['config']['op3_implementation'], sources.implementation(arm))
                self.assertEqual(worker['config']['op3_optimization_stack'], sources.STACK)
                self.assertEqual(set(worker['config']['op3_optimization_libraries']), {'ma9', 'ma11'})
                self.assertEqual((Path(worker['root'])/'worker.py').read_text(), sources.worker_source(arm))
            result = controller.read(root/'report.json')
            self.assertEqual(result['definitions'], {arm: sources.definition(arm) for arm in p.ARMS})

    def test_modified_build_is_rejected_before_next_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            calls, setup, error = self.execute_fixture(root, tamper=True)
            self.assertIn('build identity changed', str(error))
            self.assertEqual(len(calls), 1)
            self.assertEqual(controller.read(root/'batch-state.json')['status'], 'failed')

    def test_early_stops_do_not_claim_completed_benchmark(self):
        for flag, expected in (('--preflight-only', 'preflight_completed'),
                               ('--transport-smoke-only', 'transport_smoke_completed'),
                               ('--operator-only', 'operator_completed')):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                calls, setup, error = self.execute_fixture(root, stop=flag)
                self.assertIsNone(error)
                self.assertEqual(calls, [])
                state = controller.read(root/'batch-state.json')
                self.assertEqual(state['status'], expected)
                self.assertEqual(state['gpu_benchmark_completed'], flag == '--operator-only')
                self.assertFalse((root/'report.json').exists())
                if flag == '--preflight-only':
                    self.assertEqual(setup, ['hardware'])


if __name__ == '__main__':
    unittest.main()
