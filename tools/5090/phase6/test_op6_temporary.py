"""CPU regressions for the nvcc /tmp failure and whole-workflow temp routing."""
from contextlib import contextmanager, nullcontext
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import op6
import op6_child
import op6_temporary as temporary
from rtx5090_loader import load
import test_op6_runner as runner


class TemporaryStorage(unittest.TestCase):
    @contextmanager
    def broken_default(self, root):
        blocked = root / 'not-a-directory'
        blocked.write_text('unusable system temporary directory')
        with patch.dict(os.environ, {key: str(blocked) for key in temporary.VARIABLES}), \
                patch.object(tempfile, 'tempdir', str(blocked)):
            yield blocked

    def test_overrides_bad_environment_and_cached_tempdir_and_inherits_to_worker(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.broken_default(root) as blocked:
                with temporary.session(root) as directory:
                    self.assertEqual(directory.stat().st_dev, root.stat().st_dev)
                    self.assertEqual(tempfile.gettempdir(), str(directory))
                    code = '''
import json, os, tempfile
from pathlib import Path
with tempfile.NamedTemporaryFile() as f:
    print(json.dumps(dict(path=f.name, env={k:os.environ[k] for k in ('TMPDIR','TMP','TEMP')})))
'''
                    record = json.loads(subprocess.check_output([sys.executable, '-S', '-c', code],
                        env=op6.child_environment(), text=True, timeout=15))
                    self.assertEqual(Path(record['path']).parent, directory)
                    self.assertEqual(record['env'], {key: str(directory) for key in temporary.VARIABLES})
                self.assertFalse(directory.exists())
                self.assertEqual(tempfile.tempdir, str(blocked))
                self.assertTrue(all(os.environ[key] == str(blocked) for key in temporary.VARIABLES))
            record = json.loads((root/'handoff/temporary-storage.json').read_text())
            self.assertEqual(record['directory'], str(directory))

    def test_nested_parent_child_and_exception_cleanup_are_independent(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            child = root/'child'
            child.mkdir()
            with temporary.session(root) as parent_tmp:
                retained = parent_tmp/'parent-file'
                retained.write_text('keep')
                with self.assertRaisesRegex(RuntimeError, 'worker failed'):
                    with temporary.session(child) as child_tmp:
                        self.assertNotEqual(child_tmp, parent_tmp)
                        (child_tmp/'leftover').write_bytes(b'child')
                        raise RuntimeError('worker failed')
                self.assertFalse(child_tmp.exists())
                self.assertEqual(retained.read_text(), 'keep')
                self.assertEqual(tempfile.gettempdir(), str(parent_tmp))
            self.assertFalse(parent_tmp.exists())
            self.assertTrue((child/'handoff/temporary-storage.json').is_file())

    def test_long_batch_path_still_supports_multiprocessing_unix_socket(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/('long-batch-'+'x'*150)/'compare1/block03-op6'
            root.mkdir(parents=True)
            with temporary.session(root) as directory:
                self.assertLessEqual(len(os.fsencode(directory)), temporary.MAX_DIRECTORY_BYTES)
                self.assertEqual(root.stat().st_dev, directory.stat().st_dev)
                # Generate the real multiprocessing address without opening a
                # socket (the CPU test sandbox can forbid socket operations).
                code = "from multiprocessing.connection import arbitrary_address\nprint(arbitrary_address('AF_UNIX'))"
                address = subprocess.check_output([sys.executable, '-S', '-c', code], text=True, timeout=15).strip()
                self.assertTrue(address.startswith(str(directory)))
                self.assertLess(len(os.fsencode(address)), 108)

    def test_no_fallback_on_enospc_and_no_gpu_work_starts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            before = {key: os.environ.get(key) for key in temporary.VARIABLES}
            with patch.object(temporary, 'probe', side_effect=OSError(errno.ENOSPC, 'No space left')), \
                    patch.object(op6.op3.base, 'verify_controllers'), \
                    patch.object(op6, 'REPO', root), patch.object(op6, 'check_overrides'), \
                    patch.object(op6.op6_memory, 'session') as holder:
                args = op6.arguments(['--execute', '--out', str(root), '--reserve-mib', '0'])
                op6.write(root/'batch-state.json', {})
                with self.assertRaisesRegex(RuntimeError, 'No space left.*available_bytes=.*no fallback'):
                    op6.supervise(args)
                holder.assert_not_called()
            self.assertEqual(before, {key: os.environ.get(key) for key in temporary.VARIABLES})
            self.assertFalse(list(root.rglob('run-*')))
            state = op6.read(root/'batch-state.json')
            self.assertEqual((state['status'], state['stage']), ('failed', 'prepare-temporary-storage'))

    def test_existing_symlink_cannot_redirect_tempfiles_to_another_disk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            other = root/'other'
            other.mkdir()
            (root/f'.op6-tmp-{os.getuid()}').symlink_to(other, target_is_directory=True)
            with self.assertRaisesRegex(RuntimeError, 'Unsafe OP6 temporary base'):
                with temporary.session(root):
                    self.fail('symlink was accepted')
            self.assertFalse(list(other.iterdir()))

    def test_exact_build_identity_query_works_with_bad_original_tmpdir(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            nvcc = root/'cuda/bin/nvcc'
            nvcc.parent.mkdir(parents=True)
            nvcc.write_text(f'#!{sys.executable} -S\n' + '''
import os, sys, tempfile
assert sys.argv[1:] == ['--list-gpu-code']
with tempfile.NamedTemporaryFile(dir=os.environ['TMPDIR']):
    print('sm_120')
''')
            nvcc.chmod(0o700)
            torch = SimpleNamespace(__version__='fixture', version=SimpleNamespace(cuda='12.9'),
                _C=SimpleNamespace(_cuda_getArchFlags=lambda: 'sm_120', _GLIBCXX_USE_CXX11_ABI=True))
            with self.broken_default(root), patch.dict(sys.modules, torch=torch):
                with self.assertRaises(subprocess.CalledProcessError):
                    load('build').environment_identity(nvcc.parent.parent)
                with temporary.session(root):
                    identity = load('build').environment_identity(nvcc.parent.parent)
                    self.assertEqual(identity['cuda'], '12.9')

    def test_each_child_variant_installs_temp_scope_before_gpu_or_build_probes(self):
        for variant in ('op4', 'op6'):
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                out = root/'new-child'
                seen = []

                def main(_args):
                    self.assertFalse(out.exists())
                    out.mkdir()
                    op6_child.op3.supervise(SimpleNamespace(out=out, python=Path(sys.executable)))

                def supervise(_args):
                    path = Path(tempfile.gettempdir())
                    self.assertTrue(path.is_dir())
                    self.assertEqual(path.stat().st_dev, out.stat().st_dev)
                    seen.append(path)
                    self.assertEqual(os.environ['TMPDIR'], str(path))

                def reservation(*_args):
                    self.assertTrue((out/'handoff/temporary-storage.json').is_file())
                    return nullcontext(object())

                with self.broken_default(root), patch.object(op6_child, 'configured', return_value=nullcontext()), \
                        patch.object(op6_child.op3, 'main', side_effect=main), \
                        patch.object(op6_child.op3, 'supervise', side_effect=supervise), \
                        patch.object(op6_child.op6_memory, 'session', side_effect=reservation), \
                        patch.object(op6_child.op6_memory, 'child_handoff', return_value=nullcontext()):
                    op6_child.main(['--variant', variant, '--execute', '--foreground', '--out', str(out)])
                self.assertEqual(len(seen), 1)
                self.assertFalse(seen[0].exists())

    def test_all_four_paired_blocks_and_initial_build_inherit_output_temp_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            seen = []
            def inspect(_command, env, _log):
                path = Path(env['TMPDIR'])
                self.assertTrue(path.is_dir())
                self.assertEqual(path.stat().st_dev, root.stat().st_dev)
                self.assertEqual(tempfile.gettempdir(), str(path))
                seen.append(path)
            def build(*_args):
                inspect(None, op6.child_environment(), None)
            with self.broken_default(root), patch.object(op6, 'prepare_initial_build', side_effect=build):
                calls, error = runner.Supervision().run_fixture(root, blocks=4, inspect_child=inspect)
            self.assertIsNone(error)
            self.assertEqual(len(calls), 8)
            self.assertEqual(len(seen), 9)
            self.assertEqual(len(set(seen)), 1)
            self.assertFalse(seen[0].exists())


if __name__ == '__main__':
    unittest.main()
