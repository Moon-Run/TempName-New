"""MA7 launch contracts: paired blocks, immutable native code, honest outcomes."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import detach_ma7 as launch


class MA7LaunchTests(unittest.TestCase):
    def test_batch_freezes_unimported_source_templates(self):
        hashes = launch.controller_hashes()
        for name in ('ma4_f_bench.py', 'ma4_f_fullstep.py', 'ma5_fullstep.py',
                     'ma2_cr2_bench.py', 'ma7_adapter.py'):
            path = launch.HERE / name
            self.assertEqual(hashes[str(path)], launch.sha(path))

    def test_preview_does_not_create_output_or_start_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / 'absent'
            for trials, windows, steps in ((1, 16, 800), (2, 32, 1600)):
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
                self.assertIn('CR3/F4 / F4-wgrad', output.getvalue())

    def test_verified_reuse_omits_rebuild_and_preserves_all_test_stages(self):
        args = launch.arguments(['--ma7-build', '/synthetic/verified-build'])
        tasks = dict(launch.commands(args))
        self.assertNotIn('compile', tasks)
        self.assertNotIn('build-prepare', tasks)
        self.assertEqual(len(tasks), 6)
        self.assertIn(str(args.ma7_build), tasks['operator-prepare'])
        self.assertIn(str(args.ma7_build), tasks['fullstep-prepare'])
        self.assertEqual(launch.BLOCKS, 8)
        self.assertEqual(launch.ARMS, ('F4', 'F4-wgrad'))

    def test_invalid_sizes_and_stale_experiment_flags_are_rejected(self):
        for flags in (['--jobs', '0'], ['--worker-timeout', '0']):
            with self.assertRaises(ValueError):
                launch.arguments(flags)
        for flag in ('--ma6-build', '--attn-threads', '--attn-rows', '--gather-schedule'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.arguments([flag, '128'])

    def test_negative_performance_still_completes_every_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launch.arguments(['--out', temp, '--trials', '2'])
            (args.out / 'handoff').mkdir()
            (args.out / 'handoff/controllers.json').write_text('{}')
            result = dict(acceptance_status={'F4-wgrad': 'F4_increment_not_validated'},
                          increment_over_F4_validated={'F4-wgrad': False},
                          formal_optimizer_steps=1600, all_samples=[None] * 32)
            with patch.object(launch.launcher, 'invoke') as invoke, \
                 patch.object(launch, 'check_operator', return_value={}), \
                 patch.object(launch, 'check_fullstep', return_value=result) as check, \
                 contextlib.redirect_stdout(io.StringIO()):
                state = launch.supervise(args)
            self.assertEqual(invoke.call_count, 8)
            self.assertEqual(check.call_count, 2)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['stage'], 'done')
            self.assertEqual(state['timing_windows'], 32)
            self.assertEqual(state['increment_over_F4_validated'], {'F4-wgrad': False})
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


if __name__ == '__main__':
    unittest.main()
