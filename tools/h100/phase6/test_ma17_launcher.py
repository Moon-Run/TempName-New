"""MA11 launch contracts: paired blocks, immutable native code, honest outcomes."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import detach_ma17 as launch


class MA17LaunchTests(unittest.TestCase):
    def test_virtual_torch_style_modules_are_not_controller_files(self):
        dynamic = types.ModuleType('ma17_dynamic_fixture')
        def missing(name):
            raise AssertionError('Dynamic module attribute lookup: ' + name)
        dynamic.__getattr__ = missing
        virtual = types.ModuleType('ma17_virtual_fixture')
        virtual.__file__ = '_classes.py'
        with patch.dict(sys.modules, ma17_dynamic_fixture=dynamic, ma17_virtual_fixture=virtual):
            hashes = launch.controller_hashes()
        self.assertNotIn(str(launch.HERE / '_classes.py'), hashes)
        for name in ('ma17_adapter.py', 'ma17_profile.py', 'ma11_adapter.py', 'ma11_softmax.cuh'):
            self.assertEqual(hashes[str(launch.HERE / name)], launch.sha(launch.HERE / name))

    def test_batch_freezes_unimported_source_templates(self):
        hashes = launch.controller_hashes()
        for name in ('ma4_f_bench.py', 'ma4_f_fullstep.py', 'ma5_fullstep.py',
                     'ma2_cr2_bench.py', 'ma9_bench.py', 'ma8_adapter.py', 'ma9_adapter.py',
                     'ma11_adapter.py', 'ma11_softmax.cuh', 'ma17_sources.py', 'ma17_adapter.py', 'ma17_profile.py'):
            path = launch.HERE / name
            self.assertEqual(hashes[str(path)], launch.sha(path))

    def test_preview_does_not_create_output_or_start_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'absent'
            for trials, windows, steps in ((1, 8, 400), (2, 16, 800)):
                with contextlib.redirect_stdout(io.StringIO()) as output, \
                     patch.object(launch.launcher, 'invoke') as invoke, \
                     patch.object(launch.launcher, 'detach') as detach, \
                     patch.object(launch, 'controller_hashes') as hashes:
                    launch.main(['--out', str(root), '--trials', str(trials)])
                self.assertFalse(root.exists())
                invoke.assert_not_called()
                detach.assert_not_called()
                hashes.assert_not_called()
                self.assertIn(f'{windows} windows', output.getvalue())
                self.assertIn(f'{steps} formal optimizer steps', output.getvalue())
                self.assertIn('F4-cache-zcount-csoftmax / F4-cache-zcount-csoftmax-directag only', output.getvalue())
                self.assertIn('incremental over MA11', output.getvalue())
                self.assertNotIn('CR3/F4 /', output.getvalue())

    def test_verified_reuse_omits_rebuild_and_preserves_all_test_stages(self):
        args = launch.arguments(['--ma17-build', '/synthetic/verified-build'])
        tasks = dict(launch.commands(args))
        self.assertNotIn('compile', tasks)
        self.assertNotIn('build-prepare', tasks)
        self.assertEqual(len(tasks), 6)
        self.assertIn(str(args.ma17_build), tasks['operator-prepare'])
        self.assertIn(str(args.ma17_build), tasks['fullstep-prepare'])
        self.assertEqual(launch.BLOCKS, 4)
        for name in ('operator-prepare', 'fullstep-prepare'):
            self.assertEqual(tasks[name][tasks[name].index('--blocks') + 1], '4')
        self.assertEqual(launch.ARMS, ('F4-cache-zcount-csoftmax', 'F4-cache-zcount-csoftmax-directag'))

    def test_invalid_sizes_and_stale_experiment_flags_are_rejected(self):
        for flags in (['--jobs', '0'], ['--worker-timeout', '0']):
            with self.assertRaises(ValueError):
                launch.arguments(flags)
        for flag in ('--ma6-build', '--ma8-build', '--ma9-build', '--ma11-build', '--ma14-build', '--attn-threads', '--attn-rows', '--gather-schedule'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.arguments([flag, '128'])

    def test_build_jobs_default_to_nproc_and_allow_explicit_override(self):
        with patch.object(launch.subprocess, 'check_output', return_value='24\n') as nproc:
            args = launch.arguments(['--candidate-only'])
        nproc.assert_called_once_with(['nproc'], text=True)
        self.assertEqual(args.jobs, 24)
        compile_command = dict(launch.commands(args))['compile']
        self.assertEqual(compile_command[compile_command.index('--jobs')+1], '24')
        with patch.object(launch.subprocess, 'check_output', side_effect=AssertionError('unexpected nproc')):
            explicit = launch.arguments(['--jobs', '8'])
        compile_command = dict(launch.commands(explicit))['compile']
        self.assertEqual(compile_command[compile_command.index('--jobs')+1], '8')

    def test_negative_performance_still_completes_every_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launch.arguments(['--out', temp, '--trials', '2'])
            (args.out / 'handoff').mkdir()
            (args.out / 'handoff/controllers.json').write_text('{}')
            result = dict(acceptance_status={'F4-cache-zcount-csoftmax-directag': 'F4_cache_increment_not_validated'},
                          increment_over_MA11_validated={'F4-cache-zcount-csoftmax-directag': False},
                          formal_optimizer_steps=800, all_samples=[None] * 16,
                          target_evaluated=True,target_validated=False,target_point_observed=False)
            with patch.object(launch.launcher, 'invoke') as invoke, \
                 patch.object(launch, 'check_operator', return_value={}), \
                 patch.object(launch, 'check_fullstep', return_value=result) as check, \
                 contextlib.redirect_stdout(io.StringIO()):
                state = launch.supervise(args)
            self.assertEqual(invoke.call_count, 8)
            self.assertEqual(check.call_count, 2)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['stage'], 'done')
            self.assertEqual(state['timing_windows'], 16)
            self.assertEqual(state['increment_over_MA11_validated'], {'F4-cache-zcount-csoftmax-directag': False})
            self.assertEqual(json.loads((args.out / 'batch-state.json').read_text()), state)

    def test_failed_stage_retains_failure_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launch.arguments(['--out', temp])
            (args.out / 'handoff').mkdir()
            (args.out / 'handoff/controllers.json').write_text('{}')
            with patch.object(launch.launcher, 'invoke', side_effect=RuntimeError('worker failed')) as invoke:
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    launch.supervise(args)
            self.assertEqual(invoke.call_count, 1)
            state = json.loads((args.out / 'batch-state.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['stage'], 'build-prepare')
            self.assertIn('worker failed', state['error'])

    def test_positive_timing_is_not_accepted_until_profile_passes(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launch.arguments(['--out', temp])
            (args.out / 'handoff').mkdir()
            (args.out / 'handoff/controllers.json').write_text('{}')
            result = dict(acceptance_status={'F4-cache-zcount-csoftmax-directag': 'positive_F4_cache_increment'},
                increment_over_MA11_validated={'F4-cache-zcount-csoftmax-directag': True},
                formal_optimizer_steps=400, all_samples=[None] * 8,
                target_evaluated=True, target_validated=True, target_point_observed=True)
            observed = []

            def invoke(command, log, env):
                if log.name == 'fullstep-profile.log':
                    observed.append(json.loads((args.out / 'batch-state.json').read_text()))

            with patch.object(launch.launcher, 'invoke', side_effect=invoke), \
                 patch.object(launch, 'check_operator', return_value={}), \
                 patch.object(launch, 'check_fullstep', return_value=result), \
                 contextlib.redirect_stdout(io.StringIO()):
                final = launch.supervise(args)
            self.assertEqual(len(observed), 1)
            before = observed[0]
            self.assertTrue(before['timing_report_verified'])
            self.assertFalse(before['fullstep_execution_path_validated'])
            self.assertFalse(before['target_evaluated'])
            self.assertFalse(before['target_validated'])
            self.assertEqual(before['increment_over_MA11_validated'], {'F4-cache-zcount-csoftmax-directag': False})
            self.assertEqual(before['acceptance_status'], {'F4-cache-zcount-csoftmax-directag': 'pending_execution_evidence'})
            self.assertEqual(final['status'], 'completed')
            self.assertTrue(final['fullstep_execution_path_validated'])
            self.assertTrue(final['target_validated'])
            self.assertEqual(final['increment_over_MA11_validated'], {'F4-cache-zcount-csoftmax-directag': True})

    def test_profile_failure_cannot_publish_positive_timing_as_acceptance(self):
        for failure in ('profile_worker', 'profile_audit'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as temp:
                args = launch.arguments(['--out', temp])
                (args.out / 'handoff').mkdir()
                (args.out / 'handoff/controllers.json').write_text('{}')
                result = dict(acceptance_status={'F4-cache-zcount-csoftmax-directag': 'positive_F4_cache_increment'},
                    increment_over_MA11_validated={'F4-cache-zcount-csoftmax-directag': True},
                    formal_optimizer_steps=400, all_samples=[None] * 8,
                    target_evaluated=True, target_validated=True, target_point_observed=True)

                def invoke(command, log, env):
                    if failure == 'profile_worker' and log.name == 'fullstep-profile.log':
                        raise RuntimeError('profile failed')

                def check(root, trials, profile=False):
                    if failure == 'profile_audit' and profile:
                        raise RuntimeError('profile failed')
                    return result

                with patch.object(launch.launcher, 'invoke', side_effect=invoke), \
                     patch.object(launch, 'check_operator', return_value={}), \
                     patch.object(launch, 'check_fullstep', side_effect=check), \
                     self.assertRaisesRegex(RuntimeError, 'profile failed'):
                    launch.supervise(args)
                state = json.loads((args.out / 'batch-state.json').read_text())
                self.assertEqual(state['status'], 'failed')
                self.assertEqual(state['stage'], 'fullstep-profile')
                self.assertFalse(state['fullstep_execution_path_validated'])
                self.assertFalse(state['target_validated'])
                self.assertFalse(state['target_evaluated'])
                self.assertEqual(state['increment_over_MA11_validated'], {'F4-cache-zcount-csoftmax-directag': False})
                self.assertEqual(state['acceptance_status'], {'F4-cache-zcount-csoftmax-directag': 'pending_execution_evidence'})


if __name__ == '__main__':
    unittest.main()
