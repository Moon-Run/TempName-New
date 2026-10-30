"""Compile and exercise the actual host copy scheduler with a CPU CUDA-API spy.

This proves byte layout, enqueue order and ABI error handling, not GPU DMA.
The standalone real nvcc build is performed by check_ma18_native.py.
"""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest


SOURCE = Path(__file__).with_name('ma18_ipc_allgather.cu')
BYTES = 2 * 1024 * 1024


class NativeAbiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory(prefix='ma18-native-host-')
        root = Path(cls.tmp.name)
        (root / 'cuda_runtime_api.h').write_text('''#include <cstddef>
using cudaStream_t = void*;
enum cudaError_t { cudaSuccess=0, cudaErrorInvalidValue=1 };
enum cudaMemcpyKind { cudaMemcpyDeviceToDevice=3, cudaMemcpyDefault=4 };
cudaError_t cudaMemcpyAsync(void*,const void*,std::size_t,cudaMemcpyKind,cudaStream_t);
''')
        (root / 'spy.cc').write_text('''#include "cuda_runtime_api.h"
#include <cstring>
static int calls=0,fail_at=-1;
static void* destinations[32];
static const void* sources[32];
static int kinds[32];
static void* streams[32];
cudaError_t cudaMemcpyAsync(void* d,const void* s,std::size_t n,cudaMemcpyKind k,cudaStream_t st) {
  int i=calls++; destinations[i]=d; sources[i]=s; kinds[i]=k; streams[i]=st;
  if(i==fail_at) return cudaErrorInvalidValue;
  std::memcpy(d,s,n); return cudaSuccess;
}
extern "C" void reset(int fail) { calls=0; fail_at=fail; }
extern "C" int count() { return calls; }
extern "C" void* dest(int i) { return destinations[i]; }
extern "C" const void* src(int i) { return sources[i]; }
extern "C" int kind(int i) { return kinds[i]; }
extern "C" void* stream(int i) { return streams[i]; }
''')
        library = root / 'libma18_host_spy.so'
        subprocess.run(['g++', '-shared', '-fPIC', '-std=c++17', '-x', 'c++', '-I', str(root),
                        str(SOURCE), str(root / 'spy.cc'), '-o', str(library)], check=True)
        cls.lib = ctypes.CDLL(str(library))
        cls.lib.ma18_ipc_stage_cuda.argtypes = [ctypes.c_void_p, ctypes.c_void_p,
                                              ctypes.c_uint64, ctypes.c_void_p]
        cls.lib.ma18_ipc_pull_cuda.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p),
                                             ctypes.c_int, ctypes.c_uint64, ctypes.c_void_p]
        for name in ('dest', 'src', 'stream'):
            getattr(cls.lib, name).restype = ctypes.c_void_p
        cls.buffers = [(ctypes.c_ubyte * BYTES)() for _ in range(8)]
        for rank, buffer in enumerate(cls.buffers): ctypes.memset(buffer, rank + 1, BYTES)
        cls.pointers = (ctypes.c_void_p * 8)(*(ctypes.addressof(buffer) for buffer in cls.buffers))
        cls.output = (ctypes.c_ubyte * (8 * BYTES))()

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_actual_c_abi_cyclic_schedule_exact_rank_layout_and_stream(self):
        self.assertEqual(self.lib.ma18_ipc_protocol(), 18)
        for rank in range(8):
            self.lib.reset(-1)
            self.assertEqual(self.lib.ma18_ipc_pull_cuda(self.output, self.pointers, rank, BYTES, 19), 0)
            self.assertEqual(self.lib.count(), 8)
            for index in range(8):
                source = (rank + index) % 8
                self.assertEqual(self.lib.src(index), self.pointers[source])
                self.assertEqual(self.lib.dest(index), ctypes.addressof(self.output) + source * BYTES)
                self.assertEqual(self.lib.kind(index), 4)
                self.assertEqual(self.lib.stream(index), 19)
            for source in range(8):
                self.assertEqual(bytes(self.output[source * BYTES:(source + 1) * BYTES]),
                                 bytes([source + 1]) * BYTES)

    def test_stage_is_one_original_stream_d2d_copy(self):
        self.lib.reset(-1)
        self.assertEqual(self.lib.ma18_ipc_stage_cuda(self.output, self.buffers[4], BYTES, 19), 0)
        self.assertEqual(self.lib.count(), 1)
        self.assertEqual(self.lib.kind(0), 3)
        self.assertEqual(bytes(self.output[:BYTES]), bytes([5]) * BYTES)

    def test_invalid_arguments_fail_before_any_enqueue(self):
        for source, destination, size in ((None, self.output, BYTES),
                                         (self.buffers[0], None, BYTES),
                                         (self.output, self.output, BYTES),
                                         (self.buffers[0], self.output, 2)):
            self.lib.reset(-1)
            self.assertNotEqual(self.lib.ma18_ipc_stage_cuda(destination, source, size, 19), 0)
            self.assertEqual(self.lib.count(), 0)
        for rank in (-1, 8):
            self.lib.reset(-1)
            self.assertNotEqual(self.lib.ma18_ipc_pull_cuda(self.output, self.pointers, rank, BYTES, 19), 0)
            self.assertEqual(self.lib.count(), 0)
        for peers in (None, (ctypes.c_void_p * 8)(*([None] * 8))):
            self.lib.reset(-1)
            self.assertNotEqual(self.lib.ma18_ipc_pull_cuda(self.output, peers, 0, BYTES, 19), 0)
            self.assertEqual(self.lib.count(), 0)

    def test_partial_enqueue_error_stops_without_retry_or_later_copies(self):
        self.lib.reset(3)
        self.assertNotEqual(self.lib.ma18_ipc_pull_cuda(self.output, self.pointers, 4, BYTES, 19), 0)
        self.assertEqual(self.lib.count(), 4)


if __name__ == '__main__': unittest.main()
