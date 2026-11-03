"""Filesystem-only regression for full /tmp and incomplete profiler exports."""
from concurrent.futures import ThreadPoolExecutor
import errno
import gzip
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import op6_profile_io as profile_io

try:
    import torch
except ImportError:
    torch = None

TRACE = json.dumps({'traceEvents': [{'name': '真实 CUDA kernel', 'ts': 42, 'dur': 2}],
                    'deviceProperties': [{'id': 0}]}, ensure_ascii=False).encode()


def profiler(callback=None):
    return SimpleNamespace(export_chrome_trace=callback or (lambda path: Path(path).write_bytes(TRACE)))


class ProfileExport(unittest.TestCase):
    @unittest.skipIf(torch is None, 'PyTorch is required for a real CPU profiler export')
    def test_real_cpu_profiler_exports_to_the_precreated_output_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as captured:
                torch.ones(8).sum()
            dest = profile_io.export_profile(captured, root, 0)
            trace = json.loads(gzip.decompress(dest.read_bytes()))
            self.assertTrue(trace['traceEvents'])
            self.assertEqual(list(root.iterdir()), [dest])

    def test_export_uses_output_filesystem_even_if_system_temp_is_unusable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'mounted disk with spaces'
            root.mkdir()
            seen = []
            def export(path):
                path = Path(path)
                self.assertEqual(path.parent.parent, root)
                self.assertTrue(path.exists())  # Python has already probed creation.
                self.assertFalse((root/'rank0-profile.json.gz').exists())
                seen.append(path)
                path.write_bytes(TRACE)
            with patch.object(tempfile, 'gettempdir', side_effect=AssertionError('System /tmp used')):
                dest = profile_io.export_profile(profiler(export), root, 0)
            self.assertEqual(gzip.decompress(dest.read_bytes()), TRACE)
            self.assertEqual(list(root.iterdir()), [dest])
            self.assertFalse(seen[0].parent.exists())

    def test_silent_empty_missing_truncated_and_wrong_json_exports_are_rejected(self):
        callbacks = [lambda path: None,
                     lambda path: Path(path).unlink(),
                     lambda path: Path(path).write_text('{"traceEvents":['),
                     lambda path: Path(path).write_text('{}'),
                     lambda path: Path(path).write_text('{"traceEvents":[]}')]
        for callback in callbacks:
            with self.subTest(callback=callback), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                with self.assertRaisesRegex(RuntimeError, 'rank3 profile export failed.*available_bytes='):
                    profile_io.export_profile(profiler(callback), root, 3)
                self.assertEqual(list(root.iterdir()), [])

    def test_disk_full_export_retains_errno_and_never_publishes_gzip(self):
        def full(path):
            Path(path).write_text('{"traceEvents":')
            raise OSError(errno.ENOSPC, 'No space left on device', path)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(RuntimeError, 'No space left on device') as error:
                profile_io.export_profile(profiler(full), root, 1)
            self.assertEqual(error.exception.__cause__.errno, errno.ENOSPC)
            self.assertIn(str(root), str(error.exception))
            self.assertEqual(list(root.iterdir()), [])

    def test_compression_failure_never_leaves_an_empty_or_partial_final_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def partial(source, target, **_kwargs):
                target.write(source.read(12))
                raise OSError(errno.ENOSPC, 'compression disk full')
            with patch.object(profile_io.shutil, 'copyfileobj', side_effect=partial):
                with self.assertRaisesRegex(RuntimeError, 'compression disk full'):
                    profile_io.export_profile(profiler(), root, 2)
            self.assertEqual(list(root.iterdir()), [])

    def test_exported_bytes_are_not_read_wholesale_for_compression(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            original = profile_io.shutil.copyfileobj
            with patch.object(profile_io.shutil, 'copyfileobj', wraps=original) as copy:
                dest = profile_io.export_profile(profiler(), root, 5)
            self.assertEqual(copy.call_args.kwargs['length'], 1024*1024)
            self.assertEqual(gzip.decompress(dest.read_bytes()), TRACE)

    def test_eight_concurrent_ranks_use_distinct_staging_and_destinations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            def one(rank):
                payload = json.dumps({'traceEvents': [{'rank': rank}]}).encode()
                return profile_io.export_profile(profiler(lambda path: Path(path).write_bytes(payload)), root, rank)
            with ThreadPoolExecutor(max_workers=8) as pool:
                paths = list(pool.map(one, range(8)))
            self.assertEqual(len(set(paths)), 8)
            self.assertEqual(len(list(root.iterdir())), 8)
            for rank, path in enumerate(paths):
                self.assertEqual(json.loads(gzip.decompress(path.read_bytes()))['traceEvents'], [{'rank': rank}])

    def test_existing_result_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dest = root/'rank0-profile.json.gz'
            dest.write_bytes(b'existing evidence')
            with self.assertRaises(FileExistsError):
                profile_io.export_profile(profiler(), root, 0)
            self.assertEqual(dest.read_bytes(), b'existing evidence')

    def test_missing_output_directory_reports_the_actual_mount_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp)/'absent'
            with self.assertRaisesRegex(RuntimeError, 'profile export failed') as error:
                profile_io.export_profile(profiler(), missing, 0)
            self.assertIn(str(missing), str(error.exception))
            self.assertIsInstance(error.exception.__cause__, FileNotFoundError)


if __name__ == '__main__':
    unittest.main()
