"""Compile the actual helper and check exported ABI rejection without a GPU."""
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


class NativeABI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get('NVCC') or shutil.which('nvcc')
        if not compiler:
            raise unittest.SkipTest('nvcc is required for native ABI compilation')
        cls.temp = tempfile.TemporaryDirectory(prefix='ma11-native-test-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.library_path = Path(cls.temp.name) / 'helper.so'
        subprocess.run([compiler, '-x', 'cu', '-std=c++17', '-arch=sm_90',
                        '--use_fast_math', '--ftz=false', '--prec-div=true', '--prec-sqrt=true',
                        '-shared', '-Xcompiler=-fPIC',
                        str(Path(__file__).with_name('ma11_softmax.cuh')),
                        '-o', str(cls.library_path)], check=True, capture_output=True, text=True)
        cls.library = ctypes.CDLL(str(cls.library_path))
        cls.forward = cls.library.ma11_softmax_forward_cuda
        cls.backward = cls.library.ma11_softmax_backward_cuda
        cls.forward.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
        cls.backward.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
        cls.forward.restype = cls.backward.restype = ctypes.c_int

    def test_native_protocol_is_exported(self):
        query = self.library.ma11_softmax_protocol
        query.argtypes, query.restype = [], ctypes.c_int
        self.assertEqual(query(), 11)

    def test_forward_rejects_invalid_before_cuda_launch(self):
        # Synthetic addresses are used only on rejected paths; no device access.
        cases = [(None, 8192, 12288, 8192, 2048),
                 (4096, None, 12288, 8192, 2048),
                 (4096, 8192, None, 8192, 2048),
                 (4097, 8192, 12288, 8192, 2048),
                 (4096, 8192, 12289, 8192, 2048),
                 (4096, 8192, 4096, 8192, 2048),
                 (4096, 8192, 12288, 0, 2048),
                 (4096, 8192, 12288, -2048, 2048),
                 (4096, 8192, 12288, 2047, 2048),
                 (4096, 8192, 12288, 2**31, 2048),
                 (4096, 8192, 12288, 8192, 1024)]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.forward(*arguments, None), 1)  # cudaErrorInvalidValue

    def test_backward_rejects_null_unaligned_and_aliased_probability(self):
        for probability in (None, 8193, 16384):
            with self.subTest(probability=probability):
                self.assertEqual(self.backward(4096, probability, 12288, 16384,
                                               8192, 2048, None), 1)


if __name__ == '__main__':
    unittest.main()
