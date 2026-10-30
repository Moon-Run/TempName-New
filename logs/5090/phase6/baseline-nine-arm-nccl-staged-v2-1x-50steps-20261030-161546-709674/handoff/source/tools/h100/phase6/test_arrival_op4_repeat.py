"""CPU checks of op4 launch ordering and evidence gates; no fabricated GPU evidence."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import detach_arrival_op4 as batch
from common import read, write


class LauncherTests(unittest.TestCase):
    def test_preview_is_read_only(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(os, 'fork', side_effect=AssertionError('fork')), \
             patch.object(subprocess, 'run', side_effect=AssertionError('subprocess')), \
             patch.object(batch.bench, 'build_identity', side_effect=AssertionError('build lookup')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(temp)/'absent'
            batch.main(['--out', str(root)])
            self.assertFalse(root.exists())
            self.assertIn('4800 formal steps', output.getvalue())
            self.assertIn('--trials 2 --blocks 12', output.getvalue())
            self.assertIn('diagnostics only', output.getvalue())
            self.assertNotIn('--rounds', output.getvalue())

    def test_build_then_operator_evidence_then_two_training_trials_then_profile(self):
        args = batch.arguments(['--out', '/tmp/op4-plan-test', '--worker-timeout', '71'])
        tasks = batch.commands(args)
        self.assertEqual([name for name, _ in tasks], [
            'build-prepare', 'compile', 'operator-prepare', 'operator-timing',
            'operator-profile', 'fullstep-prepare', 'fullstep-run', 'fullstep-profile'])
        self.assertNotIn('--baseline', tasks[0][1])
        self.assertFalse(any('report' in cmd for _, cmd in tasks))
        train = tasks[-3][1]
        self.assertEqual(train[train.index('--trials')+1], '2')
        self.assertEqual(train[train.index('--blocks')+1], '12')
        for label, cmd in tasks:
            self.assertNotIn('--rounds', cmd)
            if '--execute' in cmd:
                self.assertEqual(cmd[cmd.index('--worker-timeout')+1], '71')
            if label in ('fullstep-run', 'fullstep-profile'):
                self.assertEqual(cmd[cmd.index('--stage')+1],
                                 'all' if label == 'fullstep-run' else 'profile')

    def test_reused_build_skips_compile_and_rejects_obsolete_scan_rounds(self):
        args = batch.arguments(['--op4-build', '/tmp/executable-op4'])
        tasks = batch.commands(args)
        self.assertEqual(tasks[0][0], 'operator-prepare')
        for _, cmd in tasks:
            if '--build' in cmd:
                self.assertEqual(cmd[cmd.index('--build')+1], str(args.op4_build))
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            batch.arguments(['--rounds', '2'])

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
            self.assertEqual(result['formal_optimizer_steps'], 4800)
            self.assertEqual(result['timing_windows'], 96)
            self.assertNotIn('arrival_benefit_validated', result)

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

    def training_fixture(self):
        plan = dict(run_id='test', trial_ids={'1': 'first', '2': 'second'})
        comparisons = {}
        for number, lower in ((1, 1.1), (2, .5)):
            estimates = {
                'F/A': dict(percent=1.2, ci95=[lower, 1.8]),
                'L/A': dict(percent=-.1, ci95=[-.3, .1]),
                'F/D': dict(percent=.1, ci95=[.01, .2]),
                'D/A': dict(percent=.5, ci95=[.3, .7]),
                'F/L': dict(percent=.4, ci95=[-.1, .9]),
            }
            comparisons[f'compare{number}'] = dict(
                run_id=plan['trial_ids'][str(number)], target_validated=lower >= 1.0,
                rounds={'combined': {'comparisons': estimates}},
                **{field: estimates[contrast]['ci95'][0] > 0
                   for field, contrast in batch.ATTRIBUTIONS.items()})
        report = dict(protocol=batch.fullstep.PROTOCOL, run_id='test', completed=True,
            correctness_passed=True, full_optimizer_steps_measured=True,
            formal_optimizer_steps=4800, timing_rank_records=768, all_samples=[{}]*96,
            comparisons=comparisons, evidence_files={}, target_validated=False,
            **{field: comparisons['compare1'][field] for field in batch.ATTRIBUTIONS})
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
                self.assertFalse(checked['lifecycle_benefit_validated'])
                self.assertTrue(checked['lifecycle_increment_validated'])
                self.assertTrue(checked['decoder_control_passed'])
                self.assertFalse(checked['decoder_increment_validated'])
                report['target_validated'] = True
                write(root/'report.json', report)
                with self.assertRaisesRegex(ValueError, 'acceptance changed'):
                    batch.fullstep_gate(root)

    def test_forged_attribution_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan, report = self.training_fixture()
            report['decoder_increment_validated'] = True
            write(root/'report.json', report)
            with patch.object(batch.fullstep, 'verify', return_value=plan), \
                 patch.object(batch.fullstep, 'report', side_effect=lambda *a, **kw: read(root/'report.json')):
                with self.assertRaisesRegex(ValueError, 'F/L attribution changed'):
                    batch.fullstep_gate(root)

    def test_profile_must_match_recomputed_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = dict(protocol=batch.fullstep.PROTOCOL, completed=True,
                           execution_validated=True, evidence_files={})
            write(root/'profile-summary.json', profile)
            with patch.object(batch.fullstep, 'profile_summary', return_value=dict(profile, completed=False)):
                with self.assertRaisesRegex(ValueError, 'differs from raw evidence'):
                    batch.check_fullstep_profile(root)


if __name__ == '__main__':
    unittest.main()
