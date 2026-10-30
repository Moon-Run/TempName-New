"""CPU-only checks of the detached op3 workflow; no simulated timings saved as evidence."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import detach_arrival_op3 as batch
from common import read, write


class LauncherTests(unittest.TestCase):
    def test_preview_has_no_side_effects(self):
        with tempfile.TemporaryDirectory() as temp, \
             patch.object(os, 'fork', side_effect=AssertionError('fork')), \
             patch.object(subprocess, 'run', side_effect=AssertionError('subprocess')), \
             patch.object(batch.bench, 'build_identity', side_effect=AssertionError('build lookup')), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            root = Path(temp)/'absent'
            batch.main(['--out', str(root)])
            self.assertFalse(root.exists())
            self.assertIn('3600 formal steps', output.getvalue())
            self.assertIn('--trials 2 --blocks 12', output.getvalue())

    def test_fresh_build_then_micro_profile_then_two_fullstep_comparisons(self):
        args = batch.arguments(['--out', '/tmp/op3-plan-test'])
        tasks = batch.commands(args)
        self.assertEqual([name for name, _ in tasks], [
            'build-prepare', 'compile', 'operator-prepare', 'operator-timing',
            'operator-profile', 'fullstep-prepare', 'fullstep-run'])
        self.assertNotIn('--baseline', tasks[0][1])
        self.assertFalse(any('report' in cmd for _, cmd in tasks))
        train = tasks[-2][1]
        self.assertEqual(train[train.index('--trials')+1], '2')
        self.assertEqual(train[train.index('--blocks')+1], '12')
        self.assertEqual(tasks[-1][1][-1], '--execute')

    def test_reuse_skips_build_and_scan_rounds_are_frozen(self):
        args = batch.arguments(['--op3-build', '/tmp/executable-op3', '--rounds', '3'])
        tasks = batch.commands(args)
        self.assertEqual(tasks[0][0], 'operator-prepare')
        for _, cmd in tasks:
            if '--rounds' in cmd:
                self.assertEqual(cmd[cmd.index('--rounds')+1], '3')
            if '--build' in cmd:
                self.assertEqual(cmd[cmd.index('--build')+1], str(args.op3_build))

    def supervise(self, root, useful=True, target=False, fail=None):
        args = batch.arguments(['--out', str(root)])
        root.mkdir()
        calls = []

        def invoke(command, log, env):
            calls.append(log.stem)
            self.assertEqual(env['NCCL_NVLS_ENABLE'], '0')
            if log.stem == fail:
                raise RuntimeError('test stage failed')
        with patch.object(batch.launcher, 'invoke', side_effect=invoke), \
             patch.object(batch, 'operator_gate', return_value=useful), \
             patch.object(batch, 'fullstep_gate', return_value=dict(
                 target_validated=target, acceptance_status='passed' if target else 'not_validated_1_percent')), \
             contextlib.redirect_stdout(io.StringIO()):
            result = batch.supervise(args)
        return calls, result

    def test_no_useful_consumption_stops_before_full_training(self):
        with tempfile.TemporaryDirectory() as temp:
            calls, result = self.supervise(Path(temp)/'batch', useful=False)
            self.assertNotIn('fullstep-prepare', calls)
            self.assertEqual(result['status'], 'completed_no_overlap')
            self.assertFalse(result['target_validated'])

    def test_below_target_completes_without_claiming_success(self):
        with tempfile.TemporaryDirectory() as temp:
            calls, result = self.supervise(Path(temp)/'batch', target=False)
            self.assertEqual(calls[-1], 'fullstep-run')
            self.assertEqual(result['status'], 'completed')
            self.assertFalse(result['target_validated'])

    def test_success_requires_both_mechanism_and_performance(self):
        with tempfile.TemporaryDirectory() as temp:
            _, result = self.supervise(Path(temp)/'batch', target=True)
            self.assertTrue(result['useful_early_consumption_validated'])
            self.assertTrue(result['target_validated'])
            self.assertEqual(result['formal_optimizer_steps'], 3600)

    def test_stage_failure_is_saved_and_stops_work(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'batch'
            with self.assertRaisesRegex(RuntimeError, 'stage failed'):
                self.supervise(root, fail='operator-timing')
            state = read(root/'batch-state.json')
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['stage'], 'operator-timing')
            self.assertFalse(state['target_validated'])

    def test_fullstep_gate_rejects_a_single_qualifying_trial(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            plan = dict(run_id='test', trial_ids={'1': 'first', '2': 'second'})
            comparisons = {}
            for number, lower in ((1, 1.1), (2, .5)):
                comparisons[f'compare{number}'] = dict(run_id=plan['trial_ids'][str(number)],
                    target_validated=lower >= 1.0,
                    rounds={'combined': {'comparisons': {'O/A': dict(percent=1.2, ci95=[lower, 1.8])}}})
            report = dict(protocol=batch.fullstep.PROTOCOL, run_id='test', completed=True,
                correctness_passed=True, full_optimizer_steps_measured=True,
                formal_optimizer_steps=3600, timing_rank_records=576, all_samples=[{}]*72,
                comparisons=comparisons, evidence_files={}, target_validated=False)
            write(root/'report.json', report)
            with patch.object(batch.fullstep, 'verify', return_value=plan):
                self.assertFalse(batch.fullstep_gate(root)['target_validated'])
                report['target_validated'] = True
                write(root/'report.json', report)
                with self.assertRaisesRegex(ValueError, 'acceptance changed'):
                    batch.fullstep_gate(root)


if __name__ == '__main__':
    unittest.main()
