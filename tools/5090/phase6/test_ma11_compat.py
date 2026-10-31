"""CPU build/provenance checks; target GPU bitwise proof is a separate probe."""
import os
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rtx5090_loader import HERE, load

compat = load('ma11_compat')


class Compat(unittest.TestCase):
    def test_probe_uses_prepared_library_and_persists_pass_or_failure(self):
        import ma11_softmax_probe as probe
        sources = load('sources')
        device = SimpleNamespace(name='NVIDIA GeForce RTX 5090', major=12, minor=0,
                                 multi_processor_count=170, uuid='CPU-FIXTURE-GPU')
        cuda = SimpleNamespace(is_available=lambda: True, device_count=lambda: 8,
                               set_device=lambda index: None, get_device_properties=lambda index: device,
                               synchronize=lambda: None)
        torch = SimpleNamespace(__version__=compat.TORCH, version=SimpleNamespace(cuda='12.9'), cuda=cuda)
        record = dict(protocol=compat.PROTOCOL, torch=compat.TORCH, library='/CPU-FIXTURE/ma11.so')
        adapter = ('class NativeSoftmax:\n'
                   '    def __init__(self, torch, library): pass\n'
                   'def native_self_test(torch, native):\n')
        cases = [('pass', "    return dict(passed=True, cases=6, forward_bitwise=True, "
                          "backward_bitwise=True, rng_unchanged=True)\n"),
                 ('fail', "    raise RuntimeError('forward_bitwise=False, backward_bitwise=True')\n")]
        with tempfile.TemporaryDirectory(prefix='rtx5090-ma11-probe-cpu-') as tmp:
            root = Path(tmp)
            (root/'ma11-compat').mkdir()
            (root/'ma11-compat/manifest.json').write_text(json.dumps(record))
            for case, body in cases:
                out = root/case
                args = probe.arguments(['--execute', '--prepared-root', str(root), '--out', str(out)])
                with self.subTest(case=case), patch.dict(sys.modules, {'torch': torch}), \
                     patch.object(compat, 'verify', return_value=record) as verify, \
                     patch.object(sources, 'softmax_adapter_source', return_value=adapter+body), \
                     patch.object(probe.ctypes, 'CDLL', return_value=object()):
                    if case == 'fail':
                        with self.assertRaisesRegex(RuntimeError, 'forward_bitwise=False'):
                            probe.run(args)
                    else:
                        probe.run(args)
                    verify.assert_called_once_with(record, root, compat.TORCH)
                result = json.loads((out/'probe-result.json').read_text())
                self.assertEqual(result['completed'], case == 'pass')
                self.assertFalse(result['performance_measured'])
                self.assertEqual(result['uuid'], device.uuid)
                if case == 'fail':
                    self.assertIn('forward_bitwise=False', result['error'])
                else:
                    self.assertTrue(result['self_test']['forward_bitwise'])
                    self.assertTrue(result['self_test']['backward_bitwise'])

    def test_probe_preview_does_not_query_gpu_or_create_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'probe'
            env = dict(os.environ, RTX5090_PYTHON=sys.executable)
            result = subprocess.run(['bash', str(HERE/'run_ma11_softmax_probe.sh'), '--out', str(out)],
                                    env=env, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn('"preview": true', result.stdout)
            self.assertFalse(out.exists())

    def test_compile_and_fail_closed_on_changed_library_or_environment(self):
        nvcc = shutil.which('nvcc')
        if not nvcc and Path('/data/apps/cuda/12.8/bin/nvcc').is_file():
            nvcc = '/data/apps/cuda/12.8/bin/nvcc'
        if not nvcc:
            self.skipTest('CUDA 12.8+ nvcc is unavailable')
        cuda = Path(nvcc).resolve().parent.parent
        codes = subprocess.check_output([str(cuda/'bin/nvcc'), '--list-gpu-code'], text=True)
        if 'sm_120' not in codes.split():
            self.skipTest('nvcc does not support SM120')
        with tempfile.TemporaryDirectory(prefix='rtx5090-ma11-compat-cpu-') as tmp:
            root = Path(tmp)
            record = compat.prepare(root, cuda, compat.TORCH)
            self.assertEqual(compat.verify(record, root, compat.TORCH), record)
            self.assertFalse(record['gpu_validated'])
            library = Path(record['library'])
            symbols = subprocess.check_output(['nm', '-D', '--defined-only', str(library)], text=True)
            self.assertIn('rtx5090_ma11_softmax', symbols)
            self.assertNotIn('_ZN12ma11_softmax', symbols)
            original = library.read_bytes()
            library.write_bytes(original+b'corrupt')
            with self.assertRaisesRegex(ValueError, 'library changed'):
                compat.verify(record, root, compat.TORCH)
            library.write_bytes(original)
            with self.assertRaisesRegex(ValueError, 'protocol/environment'):
                compat.verify(record, root, '2.6.0')
            manifest = root/'ma11-compat/manifest.json'
            manifest.write_text('{}')
            with self.assertRaisesRegex(ValueError, 'compile evidence'):
                compat.verify(record, root, compat.TORCH)


if __name__ == '__main__':
    unittest.main()
