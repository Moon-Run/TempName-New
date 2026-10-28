"""Launch contracts: three arms, immutable builds, and honest negative results."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import detach_ma6 as launch


class MA6LaunchTests(unittest.TestCase):
    def test_batch_freezes_source_templates_that_are_not_imported_modules(self):
        hashes = launch.controller_hashes()
        for name in ('ma4_f_bench.py', 'ma4_f_fullstep.py', 'ma5_fullstep.py', 'ma2_cr2_bench.py'):
            path = launch.HERE/name
            self.assertEqual(hashes[str(path)], launch.sha(path))

    def test_preview_never_creates_output_or_executes_workers(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)/'absent'
            for trials, windows, steps in ((1, 36, 1800), (2, 72, 3600)):
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
                self.assertIn('F4 / F4-mask / F4-layout', output.getvalue())

    def test_fixed_geometry_and_reuse_does_not_rebuild(self):
        args = launch.arguments([])
        self.assertEqual(launch.tuning(args), {
            role: dict(decoder_threads=128, decoder_blocks=2048, decoder_rows=1)
            for role in ('attn', 'mlp')})
        tasks = launch.commands(args)
        for _, command in tasks:
            self.assertNotIn('--gather-schedule', command)
            self.assertNotIn('--attn-rows', command)
        args.ma6_build = Path('/synthetic/already-built')
        reused = dict(launch.commands(args))
        self.assertNotIn('compile', reused)
        self.assertNotIn('build-prepare', reused)
        self.assertEqual(len(reused), 6)
        self.assertIn(str(args.ma6_build), reused['operator-prepare'])
        self.assertIn(str(args.ma6_build), reused['fullstep-prepare'])

    def test_invalid_launch_sizes_are_rejected(self):
        for flags in (['--jobs', '0'], ['--worker-timeout', '0']):
            with self.assertRaises(ValueError):
                launch.arguments(flags)

    def test_v2_tuning_and_gather_flags_are_rejected(self):
        for flag in ('--attn-threads', '--attn-blocks', '--attn-rows',
                     '--mlp-threads', '--mlp-blocks', '--mlp-rows', '--gather-schedule'):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.arguments([flag, '128'])

    def test_negative_performance_completes_every_stage(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launch.arguments(['--out', temp, '--trials', '2'])
            handoff = args.out/'handoff'
            handoff.mkdir()
            (handoff/'controllers.json').write_text('{}')
            result = dict(acceptance_status='no_positive_increment',
                          increment_over_F4_validated={'F4-mask': False, 'F4-layout': False},
                          formal_optimizer_steps=3600, all_samples=[None]*72)
            with patch.object(launch.launcher, 'invoke') as invoke, \
                 patch.object(launch, 'check_operator', return_value={}), \
                 patch.object(launch, 'check_fullstep', return_value=result) as check, \
                 contextlib.redirect_stdout(io.StringIO()):
                state = launch.supervise(args)
            self.assertEqual(invoke.call_count, 8)
            self.assertEqual(check.call_count, 2)
            self.assertEqual(state['status'], 'completed')
            self.assertEqual(state['stage'], 'done')
            self.assertEqual(state['timing_windows'], 72)
            self.assertEqual(state['increment_over_F4_validated'], result['increment_over_F4_validated'])
            self.assertEqual(json.loads((args.out/'batch-state.json').read_text()), state)

    def test_failed_stage_preserves_failure_and_stops(self):
        with tempfile.TemporaryDirectory() as temp:
            args = launch.arguments(['--out', temp])
            (args.out/'handoff').mkdir()
            (args.out/'handoff/controllers.json').write_text('{}')
            with patch.object(launch.launcher, 'invoke', side_effect=RuntimeError('worker failed')) as invoke:
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    launch.supervise(args)
            self.assertEqual(invoke.call_count, 1)
            state = json.loads((args.out/'batch-state.json').read_text())
            self.assertEqual(state['status'], 'failed')
            self.assertEqual(state['stage'], 'build-prepare')
            self.assertIn('worker failed', state['error'])


if __name__ == '__main__':
    unittest.main()
