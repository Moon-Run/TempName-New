"""Integration checks for the frozen full-step controller and launch commands."""
import argparse
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import detach_ma89 as launch
import ma89_bench as bench
import ma89_fullstep as full
from test_ma2_cr2 import fixture_plan


class FrozenIntegrationTests(unittest.TestCase):
    def test_frozen_fullstep_controller_verifies_without_repository_imports(self):
        with tempfile.TemporaryDirectory() as temp:
            parent = Path(temp)
            build = parent/'build'
            build.mkdir()
            bench.write(build/'selection-plan.json', fixture_plan()['selection'])
            root = parent/'fullstep'
            args = argparse.Namespace(root=root, build=build, megatron=parent/'megatron', trials=2, blocks=8)
            libs = {arm: dict(root='/synthetic/'+arm, files={}) for arm in bench.ARMS}
            with patch.object(bench, 'build_identity', return_value={'tuning': bench.validate_tuning(None)}), \
                 patch.object(bench, 'libraries', return_value=libs), \
                 patch.object(full, 'snapshot_megatron', return_value={}), \
                 contextlib.redirect_stdout(io.StringIO()):
                full.prepare(args)
            script = '''import ast, runpy, sys
from pathlib import Path
root = Path(sys.argv[1])
sys.path.insert(0, str(root/'scripts'))
controller = runpy.run_path(str(root/'scripts/fullstep-controller.py'))
plan = controller['verify'](root, offline=True)
assert plan['arms'] == ['F4-cache', 'F4-cache-zcount']
assert len(plan['schedule']) == 32
for name in ('adapter_source', 'worker_source', 'taco_source'):
    ast.parse(controller[name]())
assert 'ma89_zero_count_state' in controller['worker_source']()
assert 'ma89_activation_setup' in controller['worker_source']()
assert (root/'scripts/ma8_adapter.py').is_file()
assert (root/'scripts/ma9_adapter.py').is_file()
assert (root/'scripts/ma8_profile.py').is_file()
assert (root/'scripts/ma9_profile.py').is_file()
'''
            result = subprocess.run([sys.executable, '-I', '-c', script, str(root)],
                                    cwd=root/'scripts', capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_every_preview_script_exposes_its_referenced_subcommand(self):
        args = launch.arguments(['--out', '/tmp/ma89-argument-contract-only'])
        for label, command in launch.commands(args):
            action = command[2]
            result = subprocess.run([command[0], command[1], action, '--help'],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, label + ': ' + result.stderr)
            flags = {item for item in command[3:] if item.startswith('--')}
            for flag in flags:
                self.assertIn(flag, result.stdout, label + ': ' + flag)


if __name__ == '__main__':
    unittest.main()
