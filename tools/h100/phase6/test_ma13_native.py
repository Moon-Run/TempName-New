"""Real CUDA compilation/ABI checks plus an independent logical-tree model.

Host checks do not execute a GPU kernel. The launcher's full-size GPU oracle
and full-model smoke remain mandatory before accepting numerical equivalence.
"""
import ctypes
import math
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parent


def fp32(value):
    try:
        return struct.unpack('<f', struct.pack('<f', value))[0]
    except OverflowError:
        return math.copysign(math.inf, value)


def bits(value):
    return struct.pack('<f', value)


def shuffle_tree(values, maximum):
    """Model all lanes; CUDA shuffle returns the caller for out-of-range lanes."""
    values = list(values)
    for distance in (16, 8, 4, 2, 1):
        old = values[:]
        for lane, left in enumerate(old):
            right = old[lane + distance] if lane + distance < 32 else left
            values[lane] = ((right if left < right else left) if maximum
                            else fp32(left + right))
    return values[0]


def full_tree(active, maximum):
    neutral = -float.fromhex('0x1.fffffep+127') if maximum else 0.0
    logical = list(active) + [neutral] * 768
    warp_values = [shuffle_tree(logical[i:i + 32], maximum)
                   for i in range(0, 1024, 32)]
    return shuffle_tree(warp_values, maximum)


def virtual_tree(active, maximum):
    neutral = -float.fromhex('0x1.fffffep+127') if maximum else 0.0
    real_warps = [shuffle_tree(active[i:i + 32], maximum)
                  for i in range(0, 256, 32)]
    return shuffle_tree(real_warps + [neutral] * 24, maximum)


class ReductionModel(unittest.TestCase):
    def compare(self, active):
        self.assertEqual(len(active), 256)
        for maximum in (False, True):
            self.assertEqual(bits(full_tree(active, maximum)),
                             bits(virtual_tree(active, maximum)))

    def test_neutral_warps_can_be_virtualized_without_tree_change(self):
        # Test bit patterns, including signed zero, infinities and NaN. This
        # checks the reduction transformation, not host/GPU exp equivalence.
        special = [0.0, -0.0, 1.0, -1.0, fp32(2 ** -149), -fp32(2 ** -149),
                   math.inf, -math.inf, math.nan,
                   float.fromhex('0x1.fffffep+127'), -float.fromhex('0x1.fffffep+127')]
        for value in special:
            self.compare([value] * 256)
        for position in (0, 1, 31, 32, 63, 127, 128, 254, 255):
            for value in special:
                active = [0.0] * 256
                active[position] = value
                self.compare(active)

    def test_heterogeneous_float32_trees_preserve_operand_order(self):
        generator = random.Random(130026)
        for _ in range(64):
            active = [fp32(math.ldexp(generator.uniform(-1.0, 1.0),
                                    generator.randrange(-140, 120)))
                      for _ in range(256)]
            self.compare(active)

    def test_incorrect_neutral_or_shortened_tree_is_detectable(self):
        # An 8-lane shortcut can preserve a -0 that the original neutral
        # +0 additions canonicalize; dropping neutral operations is forbidden.
        active = [-0.0] * 256
        self.assertEqual(bits(full_tree(active, False)), bits(0.0))
        real = [shuffle_tree(active[i:i + 32], False) for i in range(0, 256, 32)]
        for distance in (4, 2, 1):
            old = real[:]
            real = [fp32(x + old[i + distance]) if i + distance < 8 else x
                    for i, x in enumerate(old)]
        self.assertNotEqual(bits(full_tree(active, False)), bits(real[0]))
        negative = [-10000.0] * 256
        correct = full_tree(negative, True)
        wrong = shuffle_tree([shuffle_tree(negative[i:i + 32], True)
                              for i in range(0, 256, 32)] + [0.0] * 24, True)
        self.assertNotEqual(bits(correct), bits(wrong))


class NativeABI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = os.environ.get('NVCC') or shutil.which('nvcc')
        if not compiler:
            raise unittest.SkipTest('nvcc is required for native ABI compilation')
        cls.temp = tempfile.TemporaryDirectory(prefix='ma13-native-test-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        cls.compiler = compiler
        cls.common = [compiler, '-x', 'cu', '-std=c++17', '-arch=sm_90']
        cls.precise = ['--use_fast_math', '--ftz=false', '--prec-div=true', '--prec-sqrt=true']
        cls.library_path = cls.root / 'helper.so'
        subprocess.run(cls.common + cls.precise + ['-shared', '-Xcompiler=-fPIC',
                       str(HERE / 'ma13_softmax.cuh'), '-o', str(cls.library_path)],
                       check=True, capture_output=True, text=True)
        cls.library = ctypes.CDLL(str(cls.library_path))
        cls.forward = cls.library.ma13_softmax_forward_cuda
        cls.forward.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int64] * 2 + [ctypes.c_void_p]
        cls.forward.restype = ctypes.c_int

    def test_forward_only_native_protocol_and_exports(self):
        query = self.library.ma13_softmax_protocol
        query.argtypes, query.restype = [], ctypes.c_int
        self.assertEqual(query(), 13)
        for symbol in ('ma13_softmax_backward_cuda', 'ma11_softmax_backward_cuda',
                       'ma11_softmax_forward_cuda', 'ma11_softmax_protocol'):
            with self.assertRaises(AttributeError):
                getattr(self.library, symbol)

    def test_forward_rejects_invalid_before_cuda_launch(self):
        # Addresses below are passed only to paths rejected before GPU access.
        cases = [(None, 8192, 12288, 8192, 2048),
                 (4096, None, 12288, 8192, 2048),
                 (4096, 8192, None, 8192, 2048),
                 (4097, 8192, 12288, 8192, 2048),
                 (4096, 8192, 12289, 8192, 2048),
                 (4096, 8192, 4096, 8192, 2048),
                 (4096, 8192, 12288, 0, 2048),
                 (4096, 8192, 12288, -2048, 2048),
                 (4096, 8192, 12288, 2047, 2048),
                 (4096, 8192, 12288, 2 ** 31, 2048),
                 (4096, 8192, 12288, 8192, 1024)]
        for arguments in cases:
            with self.subTest(arguments=arguments):
                self.assertEqual(self.forward(*arguments, None), 1)

    def test_compiled_exponential_work_halved_and_fastmath_restored(self):
        bodies = {}
        for label, source, flags in (
                ('reference', 'ma11_softmax.cuh', []),
                ('plain', 'ma13_softmax.cuh', []),
                ('restored', 'ma13_softmax.cuh', self.precise)):
            target = self.root / (label + '.ptx')
            subprocess.run(self.common + flags + ['-ptx', str(HERE / source), '-o', str(target)],
                           check=True, capture_output=True, text=True)
            bodies[label] = target.read_text()
        self.assertEqual(bodies['plain'], bodies['restored'])
        reference_forward = bodies['reference'].split('.visible .entry', 2)[1]
        self.assertEqual(reference_forward.count('ex2.approx.ftz.f32'), 16)
        self.assertEqual(bodies['plain'].count('ex2.approx.ftz.f32'), 8)
        self.assertNotIn('cached[4096]', bodies['plain'])
        self.assertNotRegex(bodies['plain'], r'\.(?:local)\b|\b(?:ld|st)\.local\b')


if __name__ == '__main__':
    unittest.main()
