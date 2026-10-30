"""CPU checks for the closure experiment launcher; no synthetic GPU claims."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import detach_arrival_op4_v2 as batch
from common import read, write


class LauncherTests(unittest.TestCase):
    def test_preview_is_read_only_and_reports_balanced_fullstep_work(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(os, 'fork', side_effect=AssertionError('fork')), \
             patch.object(subprocess, 'run', side_effect=AssertionError('subprocess')), \
             patch.object(batch.bench, 'build_identity', side_effect=AssertionError('build lookup')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(temp)/'absent'
            batch.main(['--out', str(root)])
            self.assertFalse(root.exists())
            self.assertIn('6400 formal steps', output.getvalue())
            self.assertIn('--trials 2 --blocks 16', output.getvalue())
            self.assertIn('phases 0..7 twice', output.getvalue())
            self.assertIn('diagnostics only', output.getvalue())
            self.assertIn('CM/SM', output.getvalue())
            self.assertNotIn('arrival_op4_build.py', output.getvalue())
            self.assertNotIn('--rounds', output.getvalue())

    def test_fresh_four_arm_build_precedes_evidence_and_complete_training(self):
        args = batch.arguments(['--out', '/tmp/op4-v2-plan-test', '--worker-timeout', '71'])
        tasks = batch.commands(args)
        self.assertEqual([name for name, _ in tasks], [
            'build-prepare', 'compile', 'operator-prepare', 'operator-timing',
            'operator-profile', 'fullstep-prepare', 'fullstep-run', 'fullstep-profile'])
        train = tasks[-3][1]
        self.assertEqual(train[train.index('--trials')+1], '2')
        self.assertEqual(train[train.index('--blocks')+1], '16')
        self.assertEqual(tasks[0][1][1], str(batch.HERE/'arrival_op4_v2_build.py'))
        for label, cmd in tasks:
            self.assertNotIn('--rounds', cmd)
            if '--execute' in cmd:
                self.assertEqual(cmd[cmd.index('--worker-timeout')+1], '71')
            if label in ('fullstep-run', 'fullstep-profile'):
                self.assertEqual(cmd[cmd.index('--stage')+1],
                                 'all' if label == 'fullstep-run' else 'profile')

    def test_reused_build_skips_compile_and_old_build_flag_is_rejected(self):
        args = batch.arguments(['--op4-v2-build', '/tmp/executable-op4-v2'])
        tasks = batch.commands(args)
        self.assertEqual(tasks[0][0], 'operator-prepare')
        for _, cmd in tasks:
            if '--build' in cmd:
                self.assertEqual(cmd[cmd.index('--build')+1], str(args.op4_v2_build))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            batch.arguments(['--op4-build', '/tmp/old-lifecycle-build'])

    def test_toolchain_and_nccl_options_reach_only_the_intended_stage(self):
        args = batch.arguments(['--build-cuda-home', '/tmp/cuda-build',
                                '--cuda-home', '/tmp/cuda-runtime', '--jobs', '3',
                                '--nccl-root', '/tmp/nccl'])
        for label, cmd in batch.commands(args):
            if label == 'compile':
                self.assertEqual(cmd[cmd.index('--cuda-home')+1], '/tmp/cuda-build')
                self.assertEqual(cmd[cmd.index('--jobs')+1], '3')
                self.assertEqual(cmd[cmd.index('--nccl-root')+1], '/tmp/nccl')
            elif '--execute' in cmd:
                self.assertEqual(cmd[cmd.index('--cuda-home')+1], '/tmp/cuda-runtime')
                self.assertNotIn('--nccl-root', cmd)

    def test_virtualenv_python_symlink_remains_the_invoked_path(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            executable = root/'venv/bin/python'
            executable.parent.mkdir(parents=True)
            executable.symlink_to(Path(sys.executable).resolve())
            out = root/'out-alias'
            out.symlink_to(root/'real-output', target_is_directory=True)
            args = batch.arguments(['--python', str(executable), '--out', str(out)])
            self.assertEqual(args.python, executable.absolute())
            self.assertNotEqual(args.python, executable.resolve())
            self.assertEqual(args.out, (root/'real-output').resolve())
            for _, command in batch.commands(args):
                self.assertEqual(command[0], str(executable.absolute()))
                if '--python' in command:
                    self.assertEqual(command[command.index('--python')+1], str(executable.absolute()))

    def supervise(self, root, target=False, fail=None, savings_ms=6.4, check_error=False):
        args = batch.arguments(['--out', str(root)])
        root.mkdir()
        calls = []

        def invoke(command, log, env):
            calls.append(log.stem)
            self.assertEqual(env['NCCL_NVLS_ENABLE'], '0')
            if log.stem == fail:
                raise subprocess.TimeoutExpired(command, 1800)
        operator = dict(operator_execution_path_validated=True,
                        precision_control_passed=True, budget_matched=True,
                        micro_savings_estimate={'mean_ms': savings_ms}, operator_comparisons={})
        training = dict(target_validated=target,
                        acceptance_status='passed' if target else 'not_validated_1_percent',
                        **{field: target for field in batch.ATTRIBUTIONS})
        with patch.object(batch.launcher, 'invoke', side_effect=invoke), \
             patch.object(batch, 'check_operator', return_value=operator,
                          side_effect=ValueError('incorrect source evidence') if check_error else None), \
             patch.object(batch, 'fullstep_gate', return_value=training), \
             patch.object(batch, 'check_fullstep_profile', return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = batch.supervise(args)
        return calls, result

    def test_low_or_negative_operator_savings_never_gate_full_training(self):
        with tempfile.TemporaryDirectory() as temp:
            for index, savings_ms in enumerate((.256, 0, -6.4)):
                with self.subTest(savings_ms=savings_ms):
                    calls, result = self.supervise(Path(temp)/f'batch{index}', savings_ms=savings_ms)
                    self.assertIn('fullstep-run', calls)
                    self.assertEqual(calls[-1], 'fullstep-profile')
                    self.assertEqual(result['status'], 'completed')
                    self.assertFalse(result['target_validated'])
                    self.assertTrue(result['fullstep_execution_path_validated'])
                    self.assertEqual(result['micro_savings_estimate']['mean_ms'], savings_ms)

    def test_training_acceptance_does_not_depend_on_operator_speedup(self):
        with tempfile.TemporaryDirectory() as temp:
            _, result = self.supervise(Path(temp)/'batch', savings_ms=-6.4, target=True)
            self.assertTrue(result['target_validated'])
            self.assertEqual(result['formal_optimizer_steps'], 6400)
            self.assertEqual(result['timing_windows'], 128)
            self.assertEqual(result['phases'], list(range(8)))
            self.assertTrue(result['precision_control_passed'])
            self.assertTrue(result['budget_matched'])
            self.assertNotIn('arrival_benefit_validated', result)
            self.assertNotIn('lifecycle_benefit_validated', result)

    def test_worker_timeout_is_recorded_and_stops_following_stages(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'batch'
            with self.assertRaises(subprocess.TimeoutExpired):
                self.supervise(root, fail='operator-timing')
            state = read(root/'batch-state.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['stage'], 'operator-timing')
            self.assertFalse(state['target_validated'])
            self.assertNotIn('fullstep', state['reports'])

    def test_invalid_operator_evidence_stops_before_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'batch'
            with self.assertRaisesRegex(ValueError, 'incorrect source evidence'):
                self.supervise(root, check_error=True)
            state = read(root/'batch-state.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['stage'], 'operator-profile')
            self.assertFalse(state['operator_execution_path_validated'])
            self.assertNotIn('fullstep', state['reports'])

    def test_fullstep_profile_failure_cannot_be_marked_completed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'batch'
            with self.assertRaises(subprocess.TimeoutExpired):
                self.supervise(root, target=True, fail='fullstep-profile')
            state = read(root/'batch-state.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['stage'], 'fullstep-profile')
            self.assertFalse(state['fullstep_execution_path_validated'])

    def operator_fixture(self):
        plan = dict(run_id='operator-fixture')
        report = dict(protocol=batch.bench.PROTOCOL, run_id=plan['run_id'], completed=True,
                      correctness_passed=True, precision_control_passed=True,
                      budget_matched=True, target_validated=False, evidence_files={},
                      micro_savings_estimate={'mean_ms': -1.0},
                      roles={role: {'comparisons': {}} for role in batch.bench.ROLES})
        profile = dict(protocol=batch.bench.PROTOCOL, run_id=plan['run_id'], completed=True,
                       execution_path_validated=True, target_validated=False, evidence_files={})
        return plan, report, profile

    def test_operator_tolerant_precision_control_does_not_require_bitwise_equality(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, report, profile = self.operator_fixture()
            write(root/'report.json', report)
            write(root/'profile-summary.json', profile)
            with patch.object(batch.bench, 'verify', return_value=plan), \
                 patch.object(batch.bench, 'report', return_value=report), \
                 patch.object(batch.bench, 'profile_summary', return_value=profile):
                result = batch.check_operator(root)
                self.assertTrue(result['precision_control_passed'])
                self.assertTrue(result['budget_matched'])

    def test_operator_requires_all_three_numerical_and_budget_controls(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for field in ('correctness_passed', 'precision_control_passed', 'budget_matched'):
                with self.subTest(field=field):
                    plan, report, profile = self.operator_fixture()
                    report[field] = False
                    write(root/'report.json', report)
                    write(root/'profile-summary.json', profile)
                    with patch.object(batch.bench, 'verify', return_value=plan), \
                         self.assertRaisesRegex(ValueError, 'correctness, precision and equal-budget'):
                        batch.check_operator(root)

    def training_fixture(self):
        plan = dict(run_id='test', trial_ids={'1': 'first', '2': 'second'})
        comparisons = {}
        for number, lower in ((1, 1.1), (2, .5)):
            estimates = {
                'CM/SM': dict(percent=1.2, ci95=[lower, 1.8]),
                'CP/SP': dict(percent=-.1, ci95=[-.3, .1]),
                'CM/CP': dict(percent=.1, ci95=[.01, .2]),
                'SM/SP': dict(percent=.4, ci95=[-.1, .9]),
            }
            comparisons[f'compare{number}'] = dict(
                run_id=plan['trial_ids'][str(number)], target_validated=lower >= 1.0,
                rounds={'combined': {'comparisons': estimates}},
                **{field: estimates[contrast]['percent'] > 0 and estimates[contrast]['ci95'][0] > 0
                   for field, contrast in batch.ATTRIBUTIONS.items()})
        report = dict(protocol=batch.fullstep.PROTOCOL, run_id='test', completed=True,
            correctness_passed=True, precision_control_passed=True, budget_matched=True,
            full_optimizer_steps_measured=True, formal_optimizer_steps=6400,
            timing_rank_records=1024, all_samples=[{}]*128,
            comparisons=comparisons, evidence_files={}, target_validated=False,
            **{field: all(c[field] for c in comparisons.values()) for field in batch.ATTRIBUTIONS})
        return plan, report

    def test_fullstep_gate_requires_both_trials_and_keeps_attribution_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, report = self.training_fixture()
            write(root/'report.json', report)
            with patch.object(batch.fullstep, 'verify', return_value=plan), \
                 patch.object(batch.fullstep, 'report', side_effect=lambda *a, **kw: read(root/'report.json')):
                checked = batch.fullstep_gate(root)
                self.assertFalse(checked['target_validated'])
                self.assertTrue(checked['mechanism_validated'])
                self.assertFalse(checked['grouping_control_passed'])
                self.assertTrue(checked['merging_benefit_validated'])
                self.assertFalse(checked['zero_reuse_control_passed'])
                report['target_validated'] = True
                write(root/'report.json', report)
                with self.assertRaisesRegex(ValueError, 'acceptance changed'):
                    batch.fullstep_gate(root)

    def test_forged_attribution_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, report = self.training_fixture()
            report['zero_reuse_control_passed'] = True
            write(root/'report.json', report)
            with patch.object(batch.fullstep, 'verify', return_value=plan), \
                 patch.object(batch.fullstep, 'report', side_effect=lambda *a, **kw: read(root/'report.json')):
                with self.assertRaisesRegex(ValueError, 'SM/SP attribution changed'):
                    batch.fullstep_gate(root)

    def test_old_op4_work_counts_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, report = self.training_fixture()
            report.update(formal_optimizer_steps=4800, timing_rank_records=768, all_samples=[{}]*96)
            write(root/'report.json', report)
            with patch.object(batch.fullstep, 'verify', return_value=plan), \
                 self.assertRaisesRegex(ValueError, '16-block'):
                batch.fullstep_gate(root)

    def test_fullstep_profile_is_bound_to_same_run_and_recomputed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = dict(protocol=batch.fullstep.PROTOCOL, run_id='trial', completed=True,
                           execution_validated=True, evidence_files={})
            write(root/'profile-summary.json', profile)
            with patch.object(batch.fullstep, 'verify', return_value={'run_id': 'trial'}), \
                 patch.object(batch.fullstep, 'profile_summary', return_value=dict(profile, completed=False)):
                with self.assertRaisesRegex(ValueError, 'differs from raw evidence'):
                    batch.check_fullstep_profile(root)
            with patch.object(batch.fullstep, 'verify', return_value={'run_id': 'another-trial'}), \
                 self.assertRaisesRegex(ValueError, 'same-run'):
                batch.check_fullstep_profile(root)


if __name__ == '__main__':
    unittest.main()
