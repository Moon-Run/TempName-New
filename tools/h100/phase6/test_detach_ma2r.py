"""CPU-only launcher and evidence gates; no CUDA work or detached children."""
import contextlib
import copy
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import detach_ma2r as batch
from common import read, write


class LauncherTests(unittest.TestCase):
    def test_preview_is_read_only_and_fresh_build_has_no_calibration_dependency(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(os, 'fork', side_effect=AssertionError('fork')), \
             patch.object(subprocess, 'run', side_effect=AssertionError('subprocess')), \
             patch.object(batch.bench, 'build_identity', side_effect=AssertionError('build lookup')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(temp)/'absent'
            batch.main(['--out', str(root)])
            self.assertFalse(root.exists())
            text = output.getvalue()
            self.assertIn('2 independent comparisons, 72 windows, 3600 formal steps', text)
            self.assertIn('--trials 2 --blocks 12', text)
            self.assertIn('R/V', text)
            self.assertNotIn('--baseline', text)
            self.assertIn('R/A point >= 1%', text)
            self.assertIn('R/A CI95 lower >= 1% is reported separately', text)

    def test_compile_and_runtime_toolkits_are_kept_separate(self):
        args = batch.arguments(['--build-cuda-home', '/tmp/build-cuda', '--cuda-home', '/tmp/run-cuda',
                                '--jobs', '3', '--worker-timeout', '81', '--nccl-root', '/tmp/nccl'])
        tasks = batch.commands(args)
        self.assertEqual([name for name, _ in tasks], ['build-prepare', 'compile', 'operator-prepare',
            'operator-timing', 'operator-profile', 'fullstep-prepare', 'fullstep-run', 'fullstep-profile'])
        for label, cmd in tasks:
            if label == 'compile':
                self.assertEqual(cmd[cmd.index('--cuda-home')+1], '/tmp/build-cuda')
                self.assertEqual(cmd[cmd.index('--jobs')+1], '3')
                self.assertIn('--nccl-root', cmd)
            elif '--execute' in cmd:
                self.assertEqual(cmd[cmd.index('--cuda-home')+1], '/tmp/run-cuda')
                self.assertEqual(cmd[cmd.index('--worker-timeout')+1], '81')
                self.assertNotIn('--nccl-root', cmd)

    def test_reuse_skips_compilation_and_preserves_python_symlink(self):
        with tempfile.TemporaryDirectory() as temp:
            python = Path(temp)/'venv/bin/python'
            python.parent.mkdir(parents=True)
            python.symlink_to(Path(sys.executable).resolve())
            args = batch.arguments(['--python', str(python), '--ma2r-build', '/tmp/ma2r-build'])
            tasks = batch.commands(args)
            self.assertEqual(tasks[0][0], 'operator-prepare')
            self.assertEqual(args.python, python.absolute())
            for _, cmd in tasks:
                self.assertEqual(cmd[0], str(python.absolute()))
                if '--build' in cmd: self.assertEqual(cmd[cmd.index('--build')+1], '/tmp/ma2r-build')

    def supervise(self, root, target=False, fail=None, invalid_operator=False):
        args = batch.arguments(['--out', str(root)])
        root.mkdir()
        calls = []
        def invoke(command, log, env):
            calls.append(log.stem)
            if log.stem == fail: raise subprocess.TimeoutExpired(command, 1800)
        operator = dict(operator_execution_path_validated=True, operator_bitwise_equal=True,
                        operator_comparisons={'diagnostic': {'percent': -20.0}})
        training = dict(target_validated=target, acceptance_status='passed' if target else 'not_validated',
                        **{field: target for field in batch.ATTRIBUTIONS})
        with patch.object(batch.launcher, 'invoke', side_effect=invoke), \
             patch.object(batch, 'check_operator', return_value=operator,
                          side_effect=ValueError('bitwise mismatch') if invalid_operator else None), \
             patch.object(batch, 'fullstep_gate', return_value=training), \
             patch.object(batch, 'check_fullstep_profile', return_value={}), \
             contextlib.redirect_stdout(io.StringIO()):
            result = batch.supervise(args)
        return calls, result

    def test_negative_operator_diagnostics_do_not_gate_full_training(self):
        with tempfile.TemporaryDirectory() as temp:
            calls, result = self.supervise(Path(temp)/'batch')
            self.assertEqual(calls[-1], 'fullstep-profile')
            self.assertIn('fullstep-run', calls)
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(result['target_validated'])
            self.assertEqual(result['formal_optimizer_steps'], 3600)

    def test_failure_stops_following_stages_and_is_saved(self):
        with tempfile.TemporaryDirectory() as temp:
            for stage in ('operator-timing', 'fullstep-profile'):
                root = Path(temp)/stage
                with self.assertRaises(subprocess.TimeoutExpired): self.supervise(root, fail=stage)
                state = read(root/'batch-state.json')
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['stage'], stage)
                self.assertFalse(state['fullstep_execution_path_validated'])

    def test_invalid_bitwise_evidence_stops_before_full_training(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'batch'
            with self.assertRaisesRegex(ValueError, 'bitwise'): self.supervise(root, invalid_operator=True)
            state = read(root/'batch-state.json')
            self.assertEqual(state['stage'], 'operator-profile')
            self.assertNotIn('fullstep', state['reports'])


class EvidenceTests(unittest.TestCase):
    def training_fixture(self, last_increment_positive=True, main=1.2, main_lower=.2):
        plan = dict(run_id='CPU-test', trials=2, blocks=12, trial_ids={'1': '1', '2': '2'})
        comparisons = {}
        for index in (1, 2):
            estimates = {'R/A': dict(percent=main, ci95=[main_lower, 2.]),
                         'R/V': dict(percent=.3, ci95=[.01, .5]),
                         'V/A': dict(percent=-.1, ci95=[-.2, .01])}
            if index == 2 and not last_increment_positive:
                estimates['R/V'] = dict(percent=.1, ci95=[-.01, .2])
            comparisons[f'compare{index}'] = dict(run_id=str(index),
                **batch.fullstep.comparison_acceptance(estimates), rounds={'combined': {'comparisons': estimates}})
        outcomes = {key: all(c[key] for c in comparisons.values()) for key in ('target_validated', *batch.ATTRIBUTIONS)}
        report = dict(protocol=batch.fullstep.PROTOCOL, run_id=plan['run_id'], completed=True,
            correctness_passed=True, identical_trajectories=True, full_optimizer_steps_measured=True,
            target_percent=1., formal_optimizer_steps=3600, timing_rank_records=576, all_samples=[{}]*72,
            comparisons=comparisons, evidence_files={}, **outcomes)
        return plan, report

    def gate(self, root, plan, report):
        write(root/'report.json', report)
        with patch.object(batch.fullstep, 'verify', return_value=plan), \
             patch.object(batch.fullstep, 'report', return_value=report):
            return batch.fullstep_gate(root)

    def test_second_trial_increment_CI_must_also_be_positive(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, report = self.training_fixture(False)
            self.assertFalse(self.gate(Path(temp), plan, report)['target_validated'])
            report['target_validated'] = True
            with self.assertRaisesRegex(ValueError, 'acceptance changed'): self.gate(Path(temp), plan, report)

    def test_point_estimate_target_does_not_require_CI_lower_one_percent(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, report = self.training_fixture()
            self.assertTrue(self.gate(Path(temp), plan, report)['target_validated'])
            self.assertFalse(report['strict_one_percent_confidence_validated'])
            self.assertFalse(report['ma2_control_passed'])

    def test_point_estimate_below_one_percent_fails_even_with_positive_CI(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, report = self.training_fixture(main=.9, main_lower=.1)
            self.assertFalse(self.gate(Path(temp), plan, report)['target_validated'])
            report['comparisons']['compare2']['one_percent_point_target_validated'] = True
            with self.assertRaisesRegex(ValueError, 'acceptance field changed'):
                self.gate(Path(temp), plan, report)

    def test_truncated_or_mismatched_trajectory_report_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            plan, original = self.training_fixture()
            for field, value in (('formal_optimizer_steps', 3599), ('identical_trajectories', False),
                                  ('comparisons', {})):
                with self.subTest(field=field):
                    report = copy.deepcopy(original)
                    report[field] = value
                    with self.assertRaisesRegex(ValueError, 'Two complete'): self.gate(Path(temp), plan, report)


if __name__ == '__main__':
    unittest.main()
