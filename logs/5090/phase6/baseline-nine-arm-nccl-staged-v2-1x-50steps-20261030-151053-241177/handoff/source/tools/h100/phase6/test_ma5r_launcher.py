"""Read-only preview, interpreter forwarding and the MA5-R experiment boundary."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import detach_ma5r as launch


class LaunchTests(unittest.TestCase):
    def test_default_is_one_split_pair_and_preview_does_not_execute(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'not-created'
            with patch.object(launch.launcher, 'invoke', side_effect=AssertionError('spawned')), \
                 patch.object(launch.launcher, 'detach', side_effect=AssertionError('forked')), \
                 contextlib.redirect_stdout(io.StringIO()) as stream:
                launch.main(['--out', str(out), '--trials', '1'])
            self.assertFalse(out.exists())
            text = stream.getvalue()
            self.assertIn('16 training windows / 800 formal steps', text)
            self.assertIn('ma5r_fullstep.py', text)
            self.assertIn('--trials 1', text)
            self.assertNotIn('--trials 2', text)

    def test_only_attention_can_be_configured(self):
        args = launch.arguments(['--attn-threads', '128', '--attn-blocks', '1024', '--trials', '2'])
        name, tuning = launch.candidates(args)[0]
        self.assertEqual(name, 'attn-t128-b1024-mlp-cr2')
        self.assertEqual(tuning['mlp']['decoder_blocks'], 2048)
        commands = launch.commands(args)
        self.assertEqual(len(commands), 8)
        self.assertIn('--attn-blocks', commands[0][2])
        self.assertEqual(commands[-1][1], 'fullstep-profile')
        for argv in (['--gemm-tile-k', '64'], ['--sweep']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                launch.arguments(argv)

    def test_source_snapshot_includes_old_decoder_dependency(self):
        hashes = launch.controller_hashes()
        for name in ('ma5r_sources.py', 'ma5_sources.py', 'ma2_c_decode.cuh',
                     'ma5r_bench.py', 'ma5r_fullstep.py', 'run_ma5_role_split.sh'):
            self.assertIn(str(launch.HERE / name), hashes)

    def test_shell_override_and_quoted_forwarding(self):
        with tempfile.TemporaryDirectory(prefix='ma5r shell ') as tmp:
            root = Path(tmp)
            tools = root / 'tools/h100/phase6'
            tools.mkdir(parents=True)
            script = tools / 'run_ma5_role_split.sh'
            script.write_text((launch.HERE / script.name).read_text())
            (tools / 'detach_ma5r.py').write_text('import sys; print(repr(sys.argv[1:]))\n')
            env = dict(os.environ, MA5R_PYTHON=sys.executable)
            result = subprocess.run(['bash', str(script), '--trials', '1', '--out', 'path with spaces'],
                                    env=env, text=True, capture_output=True, check=True)
            self.assertIn("'path with spaces'", result.stdout)


if __name__ == '__main__':
    unittest.main()
