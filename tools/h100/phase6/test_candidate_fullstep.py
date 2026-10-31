"""CPU-only checks: no hidden control jobs and no unpaired speedup claims."""
import argparse
import contextlib
import importlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import candidate_fullstep as single
from test_ma2_cr2 import fixture_plan


class CandidateTests(unittest.TestCase):
    def prepare(self, parent, experiment, trials=1):
        full = single.backend(experiment)
        build = parent / 'build'
        build.mkdir()
        full.write(build / 'selection-plan.json', fixture_plan()['selection'])
        root = parent / 'fullstep'
        args = argparse.Namespace(root=root, experiment=experiment, build=build,
                                  megatron=parent / 'megatron', trials=trials)
        libraries = {arm: dict(root='/CPU-FIXTURE/' + arm, files={}) for arm in full.bench.ARMS}
        real_verify = full.verify
        with patch.object(full.bench, 'build_identity', return_value={'tuning': full.bench.validate_tuning(None)}), \
             patch.object(full.bench, 'libraries', return_value=libraries), \
             patch.object(full, 'snapshot_megatron', return_value={}), \
             patch.object(full, 'verify', side_effect=lambda root, **kw: real_verify(root, offline=True)), \
             contextlib.redirect_stdout(io.StringIO()):
            single.prepare(args)
        return root, full

    def test_all_four_experiments_freeze_only_candidate_measurements(self):
        for experiment in single.EXPERIMENTS:
            with self.subTest(experiment=experiment), tempfile.TemporaryDirectory() as temp:
                root, full = self.prepare(Path(temp), experiment, trials=2)
                plan, template, _ = single.verify(root, offline=True)
                candidate = full.bench.ARMS[1]
                self.assertEqual(plan['arms'], [candidate])
                self.assertEqual(len(plan['schedule']), 16)
                for stage, count in [('preflight', 1), ('profile', 1), ('timing', 16), ('all', 17)]:
                    jobs = single.selected_jobs(full, template, stage)
                    self.assertEqual(len(jobs), count)
                    self.assertEqual({job['arm'] for job in jobs}, {candidate})
                self.assertIn("cpu_affinity_policy='inherit'", (root/'source/scripts/worker.py').read_text())
                code = '''import runpy,sys
from pathlib import Path
root=Path(sys.argv[1]);m=runpy.run_path(str(root/'candidate-controller.py'))
p,t,f=m['verify'](root,offline=True)
assert p['comparison_mode']=='candidate_only' and len(p['schedule'])==16
assert 'torch' not in sys.modules
'''
                done = subprocess.run([sys.executable, '-I', '-B', '-c', code, str(root)], capture_output=True, text=True)
                self.assertEqual(done.returncode, 0, done.stderr)
                plan['schedule'].append(next(full.jobs(template, 'preflight')))
                single.write(root/'plan.json', plan)
                with self.assertRaisesRegex(ValueError, 'schedule'):
                    single.verify(root, offline=True)

    def test_candidate_commands_keep_smoke_profile_and_omit_control_operator_suite(self):
        for experiment in single.EXPERIMENTS:
            launch = importlib.import_module('detach_' + experiment)
            with tempfile.TemporaryDirectory() as temp, self.subTest(experiment=experiment):
                root = Path(temp) / 'new'
                args = launch.arguments(['--candidate-only', '--out', str(root), '--trials', '2'])
                tasks = dict(launch.commands(args))
                self.assertEqual(set(tasks), {'build-prepare', 'compile', 'fullstep-prepare',
                                             'fullstep-run', 'fullstep-profile'})
                self.assertIn('all', tasks['fullstep-run'])
                self.assertIn('profile', tasks['fullstep-profile'])
                self.assertIn(experiment, tasks['fullstep-prepare'])
                with contextlib.redirect_stdout(io.StringIO()) as output:
                    launch.main(['--candidate-only', '--out', str(root), '--trials', '2'])
                self.assertFalse(root.exists())
                self.assertIn('16 windows, 800 formal optimizer steps', output.getvalue())
                self.assertIn('No MA11 control', output.getvalue())

    def test_runner_launches_only_candidate_and_preserves_original_rank_validator(self):
        with tempfile.TemporaryDirectory() as temp:
            root, full = self.prepare(Path(temp), 'ma16')
            plan, template, _ = single.verify(root, offline=True)
            launched, validated = [], []
            def worker(command, cwd, env, output, timeout):
                self.assertEqual(env['H100_MA16_ARM'], full.bench.ARMS[1])
                self.assertEqual(env['H100_MA16_EXECUTE'], '1')
                self.assertEqual(command[command.index('--nproc_per_node=8') + 1], str(root/'source/scripts/worker.py'))
                launched.append(env['E2E_MODE'])
                directory = Path(env['E2E_PROCESS_OUT'])
                for rank in range(8):
                    single.write(directory/f'rank{rank}.json', dict(rank=rank, fixture='CPU_ONLY',
                        elapsed_seconds=25., losses=[1.] * 50, grad_norms=[2.] * 50))
            def window(source, inner, job):
                self.assertEqual(job['arm'], full.bench.ARMS[1])
                validated.append(job['mode'])
                return [single.read(source/'results'/job['directory']/f'rank{rank}.json') for rank in range(8)]
            args = argparse.Namespace(root=root, stage='all', execute=True, python=Path(sys.executable),
                                      cuda_home=None, worker_timeout=10)
            with patch.object(single, 'verify', return_value=(plan, template, full)), \
                 patch.object(full.bench, 'run_worker', side_effect=worker), \
                 patch.object(full, 'window', side_effect=window), \
                 patch.object(full, 'identity', return_value=['CPU_FIXTURE'] * 8), \
                 contextlib.redirect_stdout(io.StringIO()):
                single.run(args)
            self.assertEqual(launched, ['smoke'] + ['timing'] * 8)
            self.assertIn('smoke', validated)
            result = single.read(root/'report.json')
            self.assertEqual(result['formal_optimizer_steps'], 400)
            self.assertEqual(result['timing_rank_records'], 64)
            self.assertFalse(result['target_evaluated'])
            self.assertFalse(result['paired_equivalence_evaluated'])
            self.assertNotIn('comparisons', result)
            self.assertEqual(result['trial_summaries']['trial1']['mean_ms'], 500.)
            self.assertFalse(any('smoke-' + full.bench.ARMS[0] == p.name for p in (root/'source/results').iterdir()))

    def test_failed_candidate_checks_cannot_complete_supervisor(self):
        for experiment in single.EXPERIMENTS:
            launch = importlib.import_module('detach_' + experiment)
            with tempfile.TemporaryDirectory() as temp, self.subTest(experiment=experiment):
                args = launch.arguments(['--candidate-only', '--out', temp])
                (args.out/'handoff').mkdir()
                single.write(args.out/'handoff/controllers.json', {})
                with patch.object(launch.launcher, 'invoke'), \
                     patch.object(single, 'check_complete', side_effect=ValueError('candidate validation failed')), \
                     self.assertRaisesRegex(ValueError, 'candidate validation failed'):
                    launch.supervise(args)
                self.assertEqual(single.read(args.out/'batch-state.json')['status'], 'failed')

    def test_successful_candidate_supervisor_does_not_claim_ma11_gain(self):
        for experiment in single.EXPERIMENTS:
            launch = importlib.import_module('detach_' + experiment)
            with tempfile.TemporaryDirectory() as temp, self.subTest(experiment=experiment):
                args = launch.arguments(['--candidate-only', '--out', temp])
                (args.out/'handoff').mkdir()
                single.write(args.out/'handoff/controllers.json', {})
                result = dict(acceptance_status={launch.ARMS[1]: 'candidate_measured_without_MA11_comparison'},
                              increment_over_MA11_validated={launch.ARMS[1]: False},
                              formal_optimizer_steps=400, all_samples=[{}]*8,
                              target_evaluated=False, target_validated=False, target_point_observed=False)
                with patch.object(launch.launcher, 'invoke') as invoke, \
                     patch.object(single, 'check_complete', return_value=result) as check, \
                     patch.object(launch, 'check_operator', side_effect=AssertionError('paired operator suite')), \
                     contextlib.redirect_stdout(io.StringIO()):
                    state = launch.supervise(args)
                self.assertEqual(invoke.call_count, 5)
                self.assertEqual(check.call_count, 2)
                self.assertTrue(check.call_args.kwargs['profile'])
                self.assertEqual(state['status'], 'completed')
                self.assertEqual(state['arms'], [launch.ARMS[1]])
                self.assertEqual(state['formal_optimizer_steps'], 400)
                self.assertTrue(state['fullstep_execution_path_validated'])
                self.assertFalse(state['target_evaluated'])
                self.assertIsNone(state['target_percent'])
                self.assertFalse(state['standalone_operator_executed'])


if __name__ == '__main__':
    unittest.main()
