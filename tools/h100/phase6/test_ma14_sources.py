"""MA14 native compilation, source identity and fail-closed ownership checks."""
import contextlib
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest

import ma4_f_build
import ma11_sources
import ma14_sources as source


class SourceIdentity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='ma14-source-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.reference = ma4_f_build.prepare(Path(cls.temp.name) / 'reference') / 'F4'
        cls.control = source.sources(cls.reference, source.VARIANTS[0])
        cls.candidate = source.sources(cls.reference, source.VARIANTS[1])

    def test_control_is_exact_ma11(self):
        self.assertEqual(self.control, ma11_sources.sources(
            self.reference, ma11_sources.VARIANTS[1]))

    def test_producer_decoder_and_numerics_are_not_transformed(self):
        for path in ('src/gemm_rs/epilogue_evt.hpp', source.DECODER,
                     'src/gemm_rs/ma2_c_decode.cuh', 'src/gemm_rs/taco_codec.cuh'):
            self.assertNotIn(path, self.candidate)
        for path in (ma11_sources.SOFTMAX_HEADER, ma11_sources.SOFTMAX_TU,
                     ma11_sources.SOFTMAX_LICENSE, source.ZERO_COUNT_HEADER,
                     source.CUDA_CMAKE):
            self.assertEqual(self.candidate[path], self.control[path])
        # Appending a separate launch structure preserves every byte of the
        # producer's FluxTacoConfig declaration, rather than expanding it.
        self.assertTrue(self.candidate[source.HEADER].startswith(
            (self.reference / source.HEADER).read_text()))

    def test_transformed_frontend_compiles(self):
        compile(self.candidate[source.FRONTEND], source.FRONTEND, 'exec')

    def test_one_epoch_kernel_replaces_publication_only(self):
        original = (self.reference / source.WRAPPER).read_text()
        transformed = self.candidate[source.WRAPPER]
        start = original.index('  cr2_forward(')
        end = original.index('  cr2_submission_state() const', start)
        original = original[start:end]
        start = transformed.index('  cr2_forward(')
        end = transformed.index('  cr2_submission_state() const', start)
        transformed = transformed[start:end]
        self.assertEqual(original.count('group_barrier.barrier_all(stream);'), 1)
        # The only old barrier left is strictly inside one-time allocation.
        self.assertEqual(transformed.count('group_barrier.barrier_all(stream);'), 1)
        init = transformed.index('if (ma14_ready_buffers.empty())')
        barrier = transformed.index('group_barrier.barrier_all(stream);')
        init_end = transformed.index('++ma14_initializations;', barrier)
        self.assertLess(init, barrier)
        self.assertLess(barrier, init_end)
        self.assertEqual(transformed.count('taco_ma14_epoch_barrier(publication, stream)'), 1)
        self.assertNotIn('stream_synchronize', transformed)
        self.assertIn('ma14_epoch < INT64_MAX', transformed)


class PairOwnership(unittest.TestCase):
    def setUp(self):
        self.cuda = SimpleNamespace(stream=13, device=0)
        self.cuda.current_stream = lambda: SimpleNamespace(cuda_stream=self.cuda.stream)
        self.cuda.current_device = lambda: self.cuda.device
        namespace = dict(torch=SimpleNamespace(cuda=self.cuda), threading=threading,
                         contextlib=contextlib)
        exec(compile(source.PAIR_GUARD, 'pair_guard', 'exec'), namespace)
        self.pair = namespace['_Ma14EpochPair']()
        for slot in range(2):
            self.pair.register(slot, SimpleNamespace(stream=13, device=0))

    def test_alternating_repeated_calls_and_empty_pair(self):
        for slot in [0, 1] * 12:
            with self.pair.submission(slot):
                pass
        self.pair.slots[1] = None
        with self.assertRaisesRegex(RuntimeError, 'alternating complete slots'):
            with self.pair.submission(0):
                self.fail('incomplete pair entered')

    def test_wrong_slot_rejects_and_permanently_latches_failure(self):
        with self.assertRaisesRegex(RuntimeError, 'alternating complete slots'):
            with self.pair.submission(1):
                self.fail('wrong slot entered')
        with self.assertRaisesRegex(RuntimeError, 'failed epoch pair'):
            with self.pair.submission(0):
                self.fail('failed pair entered')

    def test_cross_stream_is_rejected_before_submission(self):
        with self.pair.submission(0):
            pass
        self.cuda.stream = 21
        with self.assertRaisesRegex(RuntimeError, 'one fixed stream'):
            with self.pair.submission(1):
                self.fail('cross-stream submission entered')

    def test_cross_device_and_failed_native_submission_latch(self):
        self.cuda.device = 1
        with self.assertRaises(RuntimeError):
            with self.pair.submission(0):
                self.fail('cross-device submission entered')
        self.assertTrue(self.pair.failed)
        self.setUp()
        with self.assertRaisesRegex(ValueError, 'native failure'):
            with self.pair.submission(0):
                raise ValueError('native failure')
        self.assertTrue(self.pair.failed)
        self.assertEqual(self.pair.next_slot, 0)

    def test_registration_requires_same_stream_and_unique_slots(self):
        for slot, instance in ((0, SimpleNamespace(stream=13, device=0)),
                               (2, SimpleNamespace(stream=13, device=0))):
            with self.assertRaisesRegex(ValueError, 'two distinct slots'):
                self.pair.register(slot, instance)
        self.pair.slots[1] = None
        with self.assertRaisesRegex(ValueError, 'one creation stream'):
            self.pair.register(1, SimpleNamespace(stream=21, device=0))


class NativeEpochABI(unittest.TestCase):
    class Config(ctypes.Structure):
        _fields_ = [('peers', ctypes.c_void_p * 8),
                    ('epoch', ctypes.c_uint64), ('rank', ctypes.c_int)]

    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('nvcc')
        if compiler is None:
            raise unittest.SkipTest('nvcc is required for actual MA14 native ABI compilation')
        cls.temp = tempfile.TemporaryDirectory(prefix='ma14-native-')
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        header = root / 'gemm_rs/taco_runtime.h'
        header.parent.mkdir()
        # Exact new launch ABI; this isolated helper never depends on Torch.
        header.write_text(source.header('#pragma once\n#include <cuda_runtime_api.h>\n'))
        library = root / 'epoch.so'
        subprocess.run([compiler, '-x', 'cu', '-std=c++17', '-arch=sm_90',
                        '-shared', '-Xcompiler=-fPIC', '-I' + str(root),
                        str(Path(__file__).with_name('ma14_epoch.cuh')),
                        '-o', str(library)], check=True, capture_output=True, text=True)
        cls.library = ctypes.CDLL(str(library))
        cls.barrier = cls.library.taco_ma14_epoch_barrier
        cls.barrier.argtypes, cls.barrier.restype = [cls.Config, ctypes.c_void_p], ctypes.c_int

    def test_protocol_and_invalid_configs_without_gpu(self):
        self.assertEqual(self.library.taco_ma14_epoch_protocol(), 14)
        good = dict(peers=(ctypes.c_void_p * 8)(*[4096 * (rank + 1) for rank in range(8)]),
                    epoch=1, rank=0)
        for field, value in [('epoch', 0), ('epoch', 2**63), ('rank', -1), ('rank', 8),
                             ('peers', (ctypes.c_void_p * 8)()),
                             ('peers', (ctypes.c_void_p * 8)(*[4097] * 8))]:
            config = self.Config(**dict(good, **{field: value}))
            with self.subTest(field=field, value=value):
                self.assertEqual(self.barrier(config, None), 1)


if __name__ == '__main__':
    unittest.main()
