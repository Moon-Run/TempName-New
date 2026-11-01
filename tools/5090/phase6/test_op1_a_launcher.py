"""Side-effect-free launch and cross-arm correctness contracts."""
from contextlib import redirect_stdout
import copy
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from rtx5090_loader import HERE
import op1_a_protocol as p

spec = importlib.util.spec_from_file_location('op1_launcher_tests', HERE/'op1_a.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class Launcher(unittest.TestCase):
    def test_preview_creates_no_files_and_accepts_absent_build(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'out'
            for mode in ([], ['--preflight-only'], ['--transport-smoke-only']):
                result = subprocess.run(['bash', str(HERE/'run_op1_a.sh'), '--out', str(output),
                    '--baseline-build', str(Path(tmp)/'missing'), *mode], cwd=tmp,
                    env=dict(os.environ, RTX5090_PYTHON=sys.executable), text=True, capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                record = json.loads(result.stdout[result.stdout.index('{'):])
                self.assertEqual(record['arms'], list(p.ARMS))
                self.assertEqual(record['timing_windows'], 0 if mode else 18)
                self.assertEqual(record['formal_optimizer_steps'], 0 if mode else 900)
                self.assertFalse(output.exists())

    def test_modes_and_default_output_do_not_reuse_historical_batch(self):
        args = launcher.arguments([])
        self.assertEqual(args.out.parent, launcher.REPO.parent/'Load-5090/logs/5090/phase6')
        self.assertTrue(args.out.name.startswith('op1-a-direct-wire-v1-'))
        self.assertEqual(args.blocks, 6)
        for options in (['--report'], ['--inside'], ['--blocks', '8'], ['--execute', '--report', '--out', '/tmp/op1']):
            with self.assertRaises(ValueError):
                launcher.arguments(options)

    def test_source_failure_happens_before_gpu_or_build(self):
        import op1_a_sources as sources
        import op1_a_build as builder
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

    def test_permitted_reduction_changes_are_reported_instead_of_hidden(self):
        import op1_a_validation as validation
        rows = [dict(losses=[1], grad_norms=[2], final_parameters_sha256='a'*64,
                     parameter_trajectory=[], trainable_parameters=['weight'], smoke_output_sha256={'proj': 'b'*64})]
        first = dict(arm='flux-basic', trial=0, block=-1, mode='smoke')
        second = dict(first, arm='flux-basic-interleaved')
        with patch.object(validation, 'identity', return_value='frozen'):
            refs = {}
            identity = launcher.check_identity(rows, None, refs, first)
            launcher.check_identity(rows, identity, refs, second)
            result = launcher.numerical_comparison(refs)
            self.assertTrue(result['all_trajectories_identical'])
            self.assertFalse(result['equivalence_required'])
            for key, value in (('losses', [1.1]), ('smoke_output_sha256', {'proj': 'c'*64}),
                               ('parameter_trajectory', ['different-smoke-state'])):
                refs = {}
                launcher.check_identity(rows, None, refs, first)
                changed = copy.deepcopy(rows)
                changed[0][key] = value
                launcher.check_identity(changed, identity, refs, second)
                result = launcher.numerical_comparison(refs)
                if key == 'losses':
                    self.assertAlmostEqual(result['max_loss_abs_difference'], .1)
                    self.assertFalse(result['all_trajectories_identical'])
                elif key == 'smoke_output_sha256':
                    self.assertFalse(result['windows'][0]['smoke_projection_outputs_identical'])
                else:
                    self.assertFalse(result['all_trajectories_identical'])
            with self.assertRaisesRegex(ValueError, 'Duplicate Flux'):
                launcher.check_identity(rows, identity, refs, second)
            with self.assertRaisesRegex(ValueError, 'Incomplete paired'):
                launcher.numerical_comparison({(0, -1, 'smoke'): {'flux-basic': rows}})


if __name__ == '__main__':
    unittest.main()
