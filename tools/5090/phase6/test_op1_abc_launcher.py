"""Side-effect-free preview, numerical observations and source gates."""
import copy
from contextlib import ExitStack, redirect_stdout
import gzip
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import ModuleType
from unittest.mock import patch

from rtx5090_loader import HERE
import op1_abc as launcher
import op1_abc_protocol as p


class Launcher(unittest.TestCase):
    def test_preview_has_no_side_effects_and_accepts_absent_reuse_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'out'
            for build in ('--baseline-build', '--flux-build', '--abc-build'):
                for mode in ([], ['--preflight-only'], ['--transport-smoke-only']):
                    result = subprocess.run(['bash', str(HERE/'run_op1_abc.sh'), '--out', str(output),
                        build, str(Path(tmp)/'missing'), *mode], cwd=tmp,
                        env=dict(os.environ, RTX5090_PYTHON=sys.executable), text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    record = json.loads(result.stdout[result.stdout.index('{'):])
                    self.assertEqual(record['arms'], list(p.ARMS))
                    self.assertEqual(record['timing_windows'], 0 if mode else 12)
                    self.assertEqual(record['formal_optimizer_steps'], 0 if mode else 600)
                    self.assertFalse(record['counterbalanced_order'])
                    self.assertFalse(output.exists())

    def test_modes_and_default_output_are_isolated(self):
        args = launcher.arguments([])
        self.assertEqual(args.out.parent, launcher.REPO.parent/'Load-5090/logs/5090/phase6')
        self.assertTrue(args.out.name.startswith('op1-abc-staged-v1-'))
        self.assertEqual(args.blocks, 3)
        self.assertEqual(launcher.arguments(['--blocks', '8']).blocks, 8)
        for options in (['--report'], ['--inside'], ['--blocks', '6'],
                        ['--execute', '--report', '--out', '/tmp/op1-abc']):
            with self.assertRaises(ValueError):
                launcher.arguments(options)
        with self.assertRaises(SystemExit):
            launcher.arguments(['--flux-build', '/tmp/source', '--abc-build', '/tmp/composite'])

    def test_source_failure_happens_before_gpu_or_build(self):
        import op1_abc_sources as sources
        import op1_abc_build as builder
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            args = launcher.arguments(['--out', str(root), '--execute'])
            launcher.write(root/'batch-state.json', {})
            with patch.object(launcher, 'REPO', root), patch.object(launcher.base, 'verify_controllers'), \
                 patch.object(sources, 'worker_source', side_effect=ValueError('CPU source failure')), \
                 patch.object(launcher.base, 'hardware_preflight') as hardware, patch.object(builder, 'prepare') as build:
                with self.assertRaisesRegex(ValueError, 'CPU source failure'):
                    launcher.supervise(args)
                hardware.assert_not_called()
                build.assert_not_called()
            self.assertEqual(launcher.read(root/'batch-state.json')['status'], 'failed')

    def test_profiles_complete_before_timing_and_survive_interruption(self):
        # Execute the CPU supervisor against explicit fake GPU/build boundaries.
        # This exercises checkpoint ordering; it is not transport validation.
        with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
            root = Path(tmp)
            build = root/'build'
            build.mkdir()
            launcher.write(build/'op1-abc-build.json', {'environment': {'fixture': True}})
            launcher.write(root/'batch-state.json', {})
            args = launcher.arguments(['--out', str(root), '--abc-build', str(build), '--execute'])
            modules = {name: ModuleType(name) for name in
                       ('op1_abc_build', 'op1_abc_sources', 'op1_abc_profile', 'run', 'tp8_dp2_train')}
            fake_build, fake_sources = modules['op1_abc_build'], modules['op1_abc_sources']
            fake_build.verify = lambda *a: None
            fake_build.base_root = lambda *a: root
            for name in ('worker_source', 'adapter_source', 'taco_source'):
                setattr(fake_sources, name, lambda arm: 'pass')
            fake_sources.prepare_workers = lambda *a: dict.fromkeys(p.ARMS, {})
            modules['op1_abc_profile'].analyze = lambda events, arm, rank: dict(
                execution_validated=True, arm=arm, rank=rank)
            modules['run'].arguments = lambda *a: []
            calls = []
            def fake_run(command, cwd, env, log, timeout, peer):
                dest = Path(log).parent
                job = launcher.read(dest/'command.json')['job']
                calls.append(job)
                if job['mode'] == 'timing':
                    recorded = launcher.read(root/'profile-summary.json')
                    self.assertTrue(recorded['completed'])
                    self.assertTrue(recorded['execution_validated'])
                    raise RuntimeError('fixture timing interruption')
                if job['mode'] == 'profile':
                    for rank in range(8):
                        with gzip.open(dest/f'rank{rank}-profile.json.gz', 'wt') as handle:
                            json.dump({'traceEvents': []}, handle)
            modules['tp8_dp2_train'].run_process = fake_run
            stack.enter_context(patch.dict(sys.modules, modules))
            stack.enter_context(patch.object(launcher, 'REPO', root))
            stack.enter_context(patch.object(launcher.base, 'verify_controllers'))
            stack.enter_context(patch.object(launcher.base, 'gpu_idle'))
            stack.enter_context(patch.object(launcher.base, 'check_worker',
                side_effect=lambda root, arm, worker: root/'workers'/arm/'scripts'))
            environment = {'fixture': True}
            def hardware(args):
                record = dict(environment=environment, communication={}, cuda_visible_devices='0,1,2,3,4,5,6,7')
                launcher.write(root/'environment-preflight.json', record)
                return record
            stack.enter_context(patch.object(launcher.base, 'hardware_preflight', side_effect=hardware))
            stack.enter_context(patch.object(launcher.load('build'), 'environment_identity', return_value=environment))
            stack.enter_context(patch.object(launcher, 'native_checks'))
            stack.enter_context(patch.object(launcher, 'worker_environment', return_value={}))
            stack.enter_context(patch.object(launcher, 'window', return_value=[]))
            stack.enter_context(patch.object(launcher, 'check_identity', return_value='frozen-fixture'))
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'fixture timing interruption'):
                launcher.supervise(args)
            self.assertEqual([job['mode'] for job in calls], ['smoke']*4+['profile']*4+['timing'])
            state = launcher.read(root/'batch-state.json')
            self.assertEqual(state['status'], 'failed')
            self.assertFalse((root/'report.json').exists())
            self.assertEqual(launcher.read(root/'preflight.json'), dict(completed=True, arms=dict.fromkeys(p.ARMS, True)))

    def test_numeric_differences_are_reported_and_incomplete_pairs_rejected(self):
        rows = [dict(losses=[1], grad_norms=[2], final_parameters_sha256='a'*64,
                     parameter_trajectory=[], trainable_parameters=['weight'], smoke_output_sha256={'proj': 'b'*64})]
        group = {arm: copy.deepcopy(rows) for arm in p.ARMS}
        group[p.ARMS[0]][0]['losses'] = [1.125]
        result = launcher.numerical_comparison({(0, -1, 'smoke'): group})
        self.assertEqual(len(result), 6)
        self.assertAlmostEqual(result[p.ARMS[0]+'/flux-improved']['max_loss_abs_difference'], .125)
        self.assertFalse(result[p.ARMS[0]+'/flux-improved']['equivalence_required'])
        del group[p.ARMS[-1]]
        with self.assertRaisesRegex(ValueError, 'Incomplete four-arm'):
            launcher.numerical_comparison({(0, -1, 'smoke'): group})


if __name__ == '__main__':
    unittest.main()
