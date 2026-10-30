"""Isolated fragment-ready consumption; never edit the pinned historical A.

sources() consumes an already generated H100, dual-shape, all-remote A tree.
It preserves the scalar CTA codec and ring/BF16 arithmetic exactly.
"""
from pathlib import Path

from common import HERE, replace_once, require

HEADER = 'src/gemm_rs/taco_runtime.h'
RUNTIME = 'src/gemm_rs/taco_runtime.cu'
EPILOGUE = 'src/gemm_rs/epilogue_evt.hpp'
WRAPPER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
FRONTEND = 'python/flux/gemm_rs_taco.py'
HELPER = 'src/gemm_rs/arrival_op3.cuh'


def header(source):
    source = replace_once(source, '  int n = 0;\n};', '''  int n = 0;
  uint32_t op3_epoch = 0;
  int op3_overlap = 0;
  int op3_rounds = 2;
  int op3_diagnostics = 0;
};''')
    return source + '\n#include "gemm_rs/arrival_op3.cuh"\n'


def epilogue(source):
    source = replace_once(source,
        '      // The existing GemmRS group barrier publishes completion before DQ.',
        '''      // Publish only this callback's 16 rows after every writer is done.
      taco_op3_publish(c, dst_rank, tile_idx, step_idx);''')
    # Remote path returns above; this handles the original local BF16 stores.
    anchor = '''    }

#ifdef FLUX_TACO_BASELINE
#ifdef FLUX_TACO_SEPARATE
    CUTLASS_DEVICE void
    taco_stage_step'''
    return replace_once(source, anchor, '''#ifdef FLUX_TACO_BASELINE
      if constexpr (kTacoSupported) {
        if (params.taco.enabled)
          taco_op3_publish(params.taco, dst_rank, tile_idx, step_idx);
      }
#endif
    }

#ifdef FLUX_TACO_BASELINE
#ifdef FLUX_TACO_SEPARATE
    CUTLASS_DEVICE void
    taco_stage_step''')


def runtime(source):
    # Extract the original arithmetic body rather than maintaining a second codec.
    start = source.index('__global__ void taco_decode_ring_kernel(')
    end = source.index('\nextern "C" int taco_decode_reduce(', start)
    kernel = source[start:end]
    body = kernel[kernel.index('  const int nt = '):kernel.rindex('\n}')]
    require('flux_taco::decode(' in body and 'decode_warp' not in body,
            'Op3 requires the pinned original CTA decoder')
    decode = '''__device__ __forceinline__ void taco_op3_decode_group(
    FluxTacoConfig c, const __nv_bfloat16* local, __nv_bfloat16* output,
    int64_t group, flux_taco::Scratch& scratch) {
''' + body + '\n}\n'
    return source + '\n' + decode + r'''
extern "C" int taco_op3_configure(uint32_t epoch, int overlap, int rounds, int diagnostics) {
  if (!active.enabled || active.world != 8 || active.m != 2048 || active.n != 4096 ||
      taco_placement() != 1 || epoch == 0 || epoch > 0x7fffffffU ||
      (overlap != 0 && overlap != 1) || rounds < 1 || rounds > 4 ||
      (diagnostics != 0 && diagnostics != 1) ||
      taco_source_bytes(active) < kOp3ControlBytes)
    return int(cudaErrorInvalidValue);
  active.op3_epoch = epoch;
  active.op3_overlap = overlap;
  active.op3_rounds = rounds;
  active.op3_diagnostics = diagnostics;
  return int(cudaSuccess);
}

extern "C" int taco_op3_prepare(FluxTacoConfig c, cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch) return int(cudaErrorInvalidValue);
  // Flags and done use epochs and are NEVER cleared between invocations.
  // Diagnostic atomics/timestamps are absent from the formal timing path.
  return c.op3_diagnostics ? int(cudaMemsetAsync(taco_op3_stats(c), 0,
      kOp3StatsCount * sizeof(unsigned long long), stream)) : int(cudaSuccess);
}

// Finite opportunistic work: no CTA waits for a producer in this kernel.
// One CTA owns a fragment, i.e. the two 8-row strips produced by one callback.
__global__ void taco_op3_try_decode(FluxTacoConfig c, const __nv_bfloat16* local,
                                    __nv_bfloat16* output) {
  __shared__ flux_taco::Scratch scratch;
  __shared__ int can_decode;
  const int fragment = blockIdx.x;
  if (threadIdx.x == 0)
    can_decode = taco_op3_done(c)[fragment] != c.op3_epoch &&
                 taco_op3_is_ready(c, fragment);
  __syncthreads();  // Broadcast acquire visibility to every payload reader.
  if (!can_decode) return;
  if (c.op3_diagnostics && threadIdx.x == 0)
    taco_op3_first(taco_op3_stats(c), taco_op3_clock());
  const int tile = fragment / 8, step = fragment % 8;
  for (int compact = 0; compact < 16; ++compact) {
    const int row = (tile / 32) * 128 + step * 8 + compact % 8 + (compact / 8) * 64;
    taco_op3_decode_group(c, local, output, int64_t(row) * 32 + tile % 32, scratch);
    __syncthreads();  // Also protect the final output store before scratch reuse.
  }
  if (threadIdx.x == 0) {
    taco_op3_done(c)[fragment] = c.op3_epoch;
    if (c.op3_diagnostics) {
      const auto now = taco_op3_clock();
      taco_op3_first(taco_op3_stats(c) + 3, now);
      atomicMax(taco_op3_stats(c) + 4, now);
      atomicAdd(taco_op3_stats(c) + 2, 1ULL);
    }
  }
}

// Launched ONLY after this rank's entire GEMM and finite try stream complete.
// A rank in this kernel has no current producer left to block. Every remaining
// dependency therefore points to a different rank that is still producing;
// there is no cycle between waiting consumers and their local producers.
__global__ void taco_op3_tail_decode(FluxTacoConfig c, const __nv_bfloat16* local,
                                     __nv_bfloat16* output) {
  __shared__ flux_taco::Scratch scratch;
  const int64_t group = blockIdx.x;
  const int row = group / 32;
  const int fragment = ((row / 128) * 32 + group % 32) * 8 + (row % 64) / 8;
  if (taco_op3_done(c)[fragment] == c.op3_epoch) return;
  if (threadIdx.x == 0) {
    while (!taco_op3_is_ready(c, fragment)) __nanosleep(64);
  }
  __syncthreads();
  taco_op3_decode_group(c, local, output, group, scratch);
}

extern "C" int taco_op3_try(FluxTacoConfig c, const void* local, void* output,
                           cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch || !local || !output) return int(cudaErrorInvalidValue);
  for (int round = 0; round < c.op3_rounds; ++round) {
    taco_op3_try_decode<<<kOp3Fragments, 128, 0, stream>>>(c,
        static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
  }
  return int(cudaSuccess);
}
extern "C" int taco_op3_tail(FluxTacoConfig c, const void* local, void* output,
                            cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch || !local || !output) return int(cudaErrorInvalidValue);
  taco_op3_tail_decode<<<taco_groups(c), 128, 0, stream>>>(c,
      static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}
'''


def wrapper(source):
    source = replace_once(source, '  cudaEvent_t event_;', '''  cudaEvent_t event_;
  cudaStream_t op3_stream_ = nullptr;
  cudaEvent_t op3_input_ready_ = nullptr, op3_try_done_ = nullptr;
  torch::Tensor op3_output_;''')
    source = replace_once(source, '    CUDA_CHECK(cudaEventCreate(&event_));',
        '''    CUDA_CHECK(cudaEventCreate(&event_));
    CUDA_CHECK(cudaStreamCreateWithFlags(&op3_stream_, cudaStreamNonBlocking));
    CUDA_CHECK(cudaEventCreateWithFlags(&op3_input_ready_, cudaEventDisableTiming));
    CUDA_CHECK(cudaEventCreateWithFlags(&op3_try_done_, cudaEventDisableTiming));''')
    source = replace_once(source, '    CUDA_CHECK(cudaEventDestroy(event_));',
        '''    CUDA_CHECK(cudaStreamDestroy(op3_stream_));
    CUDA_CHECK(cudaEventDestroy(op3_input_ready_));
    CUDA_CHECK(cudaEventDestroy(op3_try_done_));
    CUDA_CHECK(cudaEventDestroy(event_));''')
    anchor = '    cutlass_op->run(args, workspace, stream);'
    source = replace_once(source, anchor, '''    if (taco.enabled) {
      TORCH_CHECK(taco.op3_epoch && taco.world == 8 && taco.m == 2048 && taco.n == 4096 &&
                      (input.size(1) == 512 || input.size(1) == 2048),
                  "Op3 requires exact TP8 attention/MLP shape and an epoch");
      TORCH_CHECK(cute::get<0>(hparams_.tile_shape()) == 128 &&
                      cute::get<1>(hparams_.tile_shape()) == 128,
                  "Op3 requires the audited 128x128 epilogue fragments");
      // Allocated on the caller's stream; its final tail/join also lives there.
      op3_output_ = torch::empty({taco.m / taco.world, taco.n}, input.options());
      CUDA_CHECK(static_cast<cudaError_t>(taco_op3_prepare(taco, stream)));
      CUDA_CHECK(cudaEventRecord(op3_input_ready_, stream));
    }
    cutlass_op->run(args, workspace, stream);
    if (taco.enabled && taco.op3_overlap) {
      // Only wait for pre-GEMM inputs/diagnostic reset, NEVER GEMM completion.
      // A dedicated stream avoids GemmV2's internal rs_stream/event handshake.
      CUDA_CHECK(cudaStreamWaitEvent(op3_stream_, op3_input_ready_, 0));
      CUDA_CHECK(static_cast<cudaError_t>(taco_op3_try(
          taco, this->output_buffer.data_ptr(), op3_output_.data_ptr(), op3_stream_)));
      CUDA_CHECK(cudaEventRecord(op3_try_done_, op3_stream_));
    }''')
    source = replace_once(source,
        '      auto output = torch::empty({taco.m / taco.world, taco.n}, buffer.options());',
        '''      TORCH_CHECK(op3_output_.defined(), "Op3 forward GEMM must precede reduction");
      auto output = op3_output_;''')
    source = replace_once(source, '''      CUDA_CHECK(static_cast<cudaError_t>(taco_decode_reduce(
          taco, buffer.data_ptr(), output.data_ptr(), stream)));''',
        '''      // Same main stream: all local producer CTAs have now finished.
      if (taco.op3_overlap)
        CUDA_CHECK(cudaStreamWaitEvent(stream, op3_try_done_, 0));
      CUDA_CHECK(static_cast<cudaError_t>(taco_op3_tail(
          taco, buffer.data_ptr(), output.data_ptr(), stream)));''')
    source = replace_once(source, '''      group_barrier.barrier_all(stream);
      return output;''', '''      group_barrier.barrier_all(stream);
      op3_output_ = torch::Tensor();
      return output;''')
    anchor = '''  forward_barrier(torch::Tensor input, torch::Tensor weight, c10::optional<torch::Tensor> bias) {
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream();'''
    return replace_once(source, anchor, anchor + '''
    // Per-fragment system acquire replaces the all-rank publication barrier.
    // The post-consumption reuse barrier in local_reduction remains intact.
    if (taco_config().enabled) return;''')


def frontend(source):
    source = replace_once(source, '        self._lock = threading.Lock()', '''        self._lock = threading.Lock()
        if (self.world, m, n) != (8, 2048, 4096) or k_local not in (512, 2048):
            raise ValueError('Op3 supports only exact TP8 attention/MLP shapes')
        self.op3_epoch = 0
        self.op3_mode = os.environ.get('H100_OP3_MODE', 'overlap')
        self.op3_rounds = int(os.environ.get('H100_OP3_ROUNDS', '2'))
        self.op3_instrument = int(os.environ.get('H100_OP3_DIAGNOSTICS', '0'))
        if (self.op3_mode not in ('serial', 'overlap') or not 1 <= self.op3_rounds <= 4 or
                self.op3_instrument not in (0, 1)):
            raise ValueError('Invalid Op3 mode/rounds/diagnostics')''')
    source = replace_once(source,
        '            self.lib.taco_configure.restype = ctypes.c_int',
        '''            self.lib.taco_configure.restype = ctypes.c_int
            self.lib.taco_op3_configure.argtypes = [ctypes.c_uint32] + [ctypes.c_int]*3
            self.lib.taco_op3_configure.restype = ctypes.c_int''')
    source = replace_once(source, '        self.pointers = (ctypes.c_void_p*self.world)(*[p.data_ptr() for p in self.peers])',
        '''        self.pointers = (ctypes.c_void_p*self.world)(*[p.data_ptr() for p in self.peers])
        # Zero readiness/done once, before any rank can start publishing epochs.
        start = self.rank * self.packet_bytes_per_source
        self.peers[self.rank][start:start+18472].zero_()
        torch.cuda.synchronize()
        dist.barrier(group=group)''')
    source = replace_once(source, '''        with self._lock:
            code = self.lib.taco_configure''', '''        with self._lock:
            if self.op3_epoch >= 0x7fffffff:
                raise RuntimeError('Op3 epoch exhausted; construct a new collective instance')
            self.op3_epoch += 1
            code = self.lib.taco_configure''')
    source = replace_once(source, '''            try:
                return self.op.forward''', '''            try:
                code = self.lib.taco_op3_configure(self.op3_epoch,
                    int(self.op3_mode == 'overlap'), self.op3_rounds, self.op3_instrument)
                if code:
                    raise RuntimeError(f'Op3 configuration rejected: CUDA error {code}')
                return self.op.forward''')
    anchor = '    @property\n    def wire_bytes_per_rank(self):'
    source = replace_once(source, anchor, '''    def op3_diagnostics(self):
        if not self.op3_instrument:
            raise RuntimeError('Enable H100_OP3_DIAGNOSTICS only for untimed profiling')
        # forward() joins the auxiliary stream before returning its queued result.
        torch.cuda.synchronize()
        start = self.rank * self.packet_bytes_per_source + 18432
        values = self.peers[self.rank][start:start+40].view(torch.int64).cpu().tolist()
        first, producer_end, count, first_end, last_end = values
        return dict(epoch=self.op3_epoch, mode=self.op3_mode, rounds=self.op3_rounds,
                    try_fragments=count, first_decode_start_ns=first,
                    first_decode_complete_ns=first_end, last_decode_complete_ns=last_end,
                    last_local_publish_ns=producer_end,
                    useful_overlap=0 < first_end < producer_end)

''' + anchor)
    return source


def sources(root):
    root = Path(root)
    return {path: transform((root/path).read_text()) for path, transform in (
        (HEADER, header), (RUNTIME, runtime), (EPILOGUE, epilogue),
        (WRAPPER, wrapper), (FRONTEND, frontend))} | {
            HELPER: (HERE/'arrival_op3_cuda.cuh').read_text()}
