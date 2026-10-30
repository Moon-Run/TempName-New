"""MA14 native compilation, source identity and fail-closed ownership checks."""
import ast
import ctypes
from pathlib import Path
import shutil
import subprocess
import tempfile
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

    def test_python_hotpath_is_exact_ma11_and_native_bind_runs_once(self):
        original = ast.parse((self.reference / source.FRONTEND).read_text())
        candidate = ast.parse(self.candidate[source.FRONTEND])
        def methods(tree):
            return {(cls.name, method.name): ast.dump(method)
                    for cls in tree.body if isinstance(cls, ast.ClassDef)
                    for method in cls.body if isinstance(method, ast.FunctionDef)
                    and method.name in ('forward', 'forward_consumer')}
        self.assertEqual(methods(original), methods(candidate))
        self.assertEqual(self.candidate[source.FRONTEND].count(
            'self.slots[0].op.ma14_bind_pair(self.slots[1].op)'), 1)
        self.assertNotIn('_Ma14EpochPair', self.candidate[source.FRONTEND])
        self.assertNotIn('contextlib', self.candidate[source.FRONTEND])

    def test_both_native_entrypoints_guard_validation_and_auxiliary_rejects(self):
        native = self.candidate[source.WRAPPER]
        public = native[native.index('GemmRS::~GemmRS()'):]
        self.assertEqual(public.count('return impl_->ma14_submit([&]'), 2)
        self.assertEqual(public.count('impl_->ma14_reject_auxiliary();'), 4)
        self.assertIn('if (ma14_pair) ma14_pair->detach(ma14_slot);', native)
        self.assertIn('taco_ma14::Submission submission(ma14_pair.get(), ma14_slot);', native)
        self.assertIn('current == taco_stream', native)
        self.assertIn('input.device().index() == taco_stream.device_index()', native)
        self.assertIn('cr2_calls == 0 && other.cr2_calls == 0', native)
        self.assertIn('this != &other', native)

    def test_cached_peer_pointers_are_initialized_once(self):
        native = self.candidate[source.WRAPPER]
        begin = native.index('if (ma14_ready_buffers.empty())')
        end = native.index('++ma14_initializations;', begin)
        self.assertIn('ma14_publication.peers[peer] =', native[begin:end])
        self.assertEqual(native.count('ma14_publication.peers[peer] ='), 1)
        self.assertNotIn('Ma14EpochConfig publication;', native)

    def test_actual_generated_submit_helper_latches_before_gpu_work(self):
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            self.skipTest('C++17 compiler required for generated submission guard')
        native = self.candidate[source.WRAPPER]
        begin = native.index('  template <typename Submit>\n  torch::Tensor ma14_submit(')
        end = native.index('  void ma14_reject_auxiliary()', begin)
        helper = native[begin:end]
        program = r'''
#include "ma14_pair.h"
#include <cassert>
#include <memory>
namespace torch { using Tensor = int; }
struct Instance {
  struct { bool enabled = true; } owned_taco;
  std::shared_ptr<taco_ma14::PairState> ma14_pair;
  std::recursive_mutex ma4_consumer_mutex;
  int ma14_slot = -1;
  bool cr2_failed = false;
HELPER
};
template <typename F> void fails(F action) {
  bool rejected = false;
  try { action(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
}
int main() {
  int gpu_work = 0;
  Instance unbound;
  fails([&] { unbound.ma14_submit([&] { ++gpu_work; return 1; }); });
  assert(unbound.cr2_failed && gpu_work == 0);
  auto pair = std::make_shared<taco_ma14::PairState>(
      taco_ma14::Endpoint{0, 0, 4096}, taco_ma14::Endpoint{0, 0, 8192});
  Instance first, second;
  first.ma14_pair = second.ma14_pair = pair;
  first.ma14_slot = 0; second.ma14_slot = 1;
  fails([&] { first.ma14_submit([&]() -> int {
    throw std::runtime_error("existing native stream/layout check failed");
  }); });
  assert(first.cr2_failed && pair->failed && gpu_work == 0);
  fails([&] { second.ma14_submit([&] { ++gpu_work; return 1; }); });
  assert(second.cr2_failed && gpu_work == 0);
  Instance generic;
  generic.owned_taco.enabled = false;
  assert(generic.ma14_submit([&] { ++gpu_work; return 17; }) == 17);
  assert(gpu_work == 1 && !generic.cr2_failed);
}
'''.replace('HELPER', helper)
        root = Path(self.temp.name)
        test, binary = root / 'actual-submit.cc', root / 'actual-submit'
        test.write_text(program)
        subprocess.run([compiler, '-std=c++17', '-O2', '-pthread',
                        '-I' + str(Path(__file__).resolve().parent), str(test), '-o', str(binary)],
                       check=True, capture_output=True, text=True)
        subprocess.run([str(binary)], check=True, capture_output=True, text=True, timeout=5)

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
        self.assertEqual(transformed.count('taco_ma14_epoch_barrier(ma14_publication, stream)'), 1)
        self.assertNotIn('stream_synchronize', transformed)
        self.assertIn('ma14_epoch < INT64_MAX', transformed)


class NativePairOwnership(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        compiler = shutil.which('c++') or shutil.which('g++')
        if compiler is None:
            raise unittest.SkipTest('C++17 compiler required for actual native guard tests')
        cls.temp = tempfile.TemporaryDirectory(prefix='ma14-native-pair-')
        cls.addClassCleanup(cls.temp.cleanup)
        root = Path(cls.temp.name)
        test = root / 'pair.cc'
        test.write_text(PAIR_TEST)
        cls.binary = root / 'pair'
        subprocess.run([compiler, '-std=c++17', '-O2', '-pthread',
                        '-I' + str(Path(__file__).resolve().parent),
                        str(test), '-o', str(cls.binary)],
                       check=True, capture_output=True, text=True)

    def run_case(self, case):
        subprocess.run([str(self.binary), case], check=True, capture_output=True,
                       text=True, timeout=5)

    def test_successful_alternation(self):
        self.run_case('alternate')

    def test_wrong_slot_poisoned(self):
        self.run_case('wrong')

    def test_exception_poisons_other_slot(self):
        self.run_case('error')

    def test_destructed_endpoint_prevents_reuse_without_dangling_object(self):
        self.run_case('detach')

    def test_reentry_fails_without_deadlock(self):
        self.run_case('reentry')

    def test_double_completion_fails(self):
        self.run_case('double')

    def test_binding_stream_device_and_workspace_identity(self):
        for case in ('stream', 'device', 'alias', 'null_packet'):
            with self.subTest(case=case):
                self.run_case(case)

    def test_unbound_forward_is_rejected(self):
        self.run_case('unbound')

    def test_concurrent_calls_serialize(self):
        self.run_case('threads')


PAIR_TEST = r'''
#include "ma14_pair.h"
#include <atomic>
#include <cassert>
#include <chrono>
#include <memory>
#include <string>
#include <thread>
using namespace taco_ma14;

template <typename F> void fails(F action) {
  bool rejected = false;
  try { action(); } catch (const std::runtime_error&) { rejected = true; }
  assert(rejected);
}

int main(int argc, char** argv) {
  assert(argc == 2);
  const std::string test = argv[1];
  Endpoint a{0, 0, 4096}, b{0, 0, 8192};
  if (test == "stream" || test == "device" || test == "alias" || test == "null_packet") {
    if (test == "stream") b.stream = 7;
    if (test == "device") b.device = 1;
    if (test == "alias") b.packet = a.packet;
    if (test == "null_packet") b.packet = 0;
    fails([&] { PairState rejected(a, b); });
    return 0;
  }
  if (test == "unbound") {
    fails([] { Submission rejected(nullptr, 0); });
    return 0;
  }
  auto first = std::make_shared<PairState>(a, b);
  auto second = first;
  PairState& pair = *first;
  if (test == "alternate") {
    for (int call = 0; call < 200; ++call) {
      Submission guard(first.get(), call % 2);
      assert(pair.next_slot == call % 2 && pair.active && !pair.failed);
      guard.complete();
    }
    assert(pair.next_slot == 0 && !pair.active && !pair.failed);
  } else if (test == "wrong") {
    fails([&] { Submission rejected(first.get(), 1); });
    assert(pair.failed && pair.next_slot == 0);
    fails([&] { Submission rejected(second.get(), 0); });
  } else if (test == "error") {
    fails([&] { Submission active(first.get(), 0); throw std::runtime_error("native failure"); });
    assert(pair.failed && !pair.active && pair.next_slot == 0);
    fails([&] { Submission rejected(second.get(), 1); });
  } else if (test == "detach") {
    { Submission active(first.get(), 0); active.complete(); }
    first->detach(0);
    first.reset();
    assert(second->failed && !second->attached[0]);
    fails([&] { Submission rejected(second.get(), 1); });
  } else if (test == "reentry") {
    Submission active(first.get(), 0);
    fails([&] { Submission rejected(second.get(), 0); });
    assert(pair.failed);
    fails([&] { active.complete(); });
  } else if (test == "double") {
    Submission active(first.get(), 0);
    active.complete();
    fails([&] { active.complete(); });
    assert(pair.next_slot == 1 && pair.failed);
  } else if (test == "threads") {
    std::atomic<bool> started{false}, entered{false};
    std::thread other;
    {
      Submission active(first.get(), 0);
      other = std::thread([&] {
        started = true;
        Submission pending(second.get(), 1);
        entered = true;
        pending.complete();
      });
      while (!started) std::this_thread::yield();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      assert(!entered);
      active.complete();
    }
    other.join();
    assert(entered && pair.next_slot == 0 && !pair.failed && !pair.active);
  } else { assert(false); }
}
'''


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
        self.assertEqual(self.library.taco_ma14_epoch_protocol(), source.NATIVE_PROTOCOL)
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
