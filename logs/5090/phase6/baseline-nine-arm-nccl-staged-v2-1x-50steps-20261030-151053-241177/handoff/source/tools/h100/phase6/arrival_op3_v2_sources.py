"""Independent op3-v2 transform of pinned A; old op3 remains reproducible.

D changes only the decoder execution layout. O also enables low-cost fragment
publication and a finite, capped early consumer. S uses the O binary in serial mode.
"""
from pathlib import Path

import arrival_op3_sources as legacy
from common import HERE, replace_once

HEADER, RUNTIME, EPILOGUE = legacy.HEADER, legacy.RUNTIME, legacy.EPILOGUE
WRAPPER, FRONTEND = legacy.WRAPPER, legacy.FRONTEND
HELPER = 'src/gemm_rs/arrival_op3_v2.cuh'
DECODER = 'src/gemm_rs/arrival_op3_v2_decode.cuh'


def header(source):
    return legacy.header(source).replace('"gemm_rs/arrival_op3.cuh"',
                                         '"gemm_rs/arrival_op3_v2.cuh"')


def epilogue(source):
    text = legacy.epilogue(source)
    # publish itself supplies the CTA barrier; keep exactly one after encoding.
    return replace_once(text, '''      __syncthreads();
      // Publish only this callback's 16 rows after every writer is done.
      taco_op3_publish(c, dst_rank, tile_idx, step_idx);''', '''      // One collective barrier followed by one cumulative system release.
      taco_op3_publish(c, dst_rank, tile_idx, step_idx);''')


def decoder_only_runtime(source):
    text = replace_once(source, '#include "gemm_rs/taco_codec.cuh"',
        '#include "gemm_rs/taco_codec.cuh"\n#include "gemm_rs/arrival_op3_v2_decode.cuh"')
    anchor = 'extern "C" int taco_decode_reduce('
    kernel = '''__global__ void taco_op3_v2_decode(FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {
  const int64_t group = int64_t(blockIdx.x) * 4 + threadIdx.x / 32;
  op3_v2_decode_warp(c, local, output, group);
}

'''
    text = replace_once(text, anchor, kernel+anchor)
    return replace_once(text, '''  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  taco_decode_ring_kernel<<<taco_groups(c), 128, 0, stream>>>(''', '''  if (!c.enabled || !local || !output || c.world != 8 || c.m != 2048 || c.n != 4096)
    return int(cudaErrorInvalidValue);
  taco_op3_v2_decode<<<taco_groups(c) / 4, 128, 0, stream>>>(''')


def runtime(source):
    text = replace_once(source, '#include "gemm_rs/taco_codec.cuh"',
        '#include "gemm_rs/taco_codec.cuh"\n#include "gemm_rs/arrival_op3_v2_decode.cuh"')
    return text + r'''
extern "C" int taco_op3_protocol_version() { return 2; }
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
  // ready, group-done and stop all use epochs: no per-call control reset race.
  return c.op3_diagnostics ? int(cudaMemsetAsync(taco_op3_stats(c), 0,
      kOp3StatsCount * sizeof(unsigned long long), stream)) : int(cudaSuccess);
}
__global__ void taco_op3_v2_stop(FluxTacoConfig c) {
  auto* p = taco_op3_stop_ptr(c);
  asm volatile("st.release.gpu.global.u32 [%0], %1;" :: "l"(p), "r"(c.op3_epoch) : "memory");
  if (c.op3_diagnostics) taco_op3_stats(c)[5] = taco_op3_clock();
}
extern "C" int taco_op3_stop(FluxTacoConfig c, cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch) return int(cudaErrorInvalidValue);
  taco_op3_v2_stop<<<1, 1, 0, stream>>>(c);
  return int(cudaGetLastError());
}

// At most the first 64 physical fragments are eligible (1024 H128 groups).
// Every round visits the SAME set; group-done prevents expanding the budget.
// Four warps decode four rows concurrently, with at most four rows per warp.
// There is no spin on producers. Late CTAs and later rows stop taking work as
// soon as the main stream's post-GEMM stop epoch is observed.
__global__ void taco_op3_v2_try_decode(FluxTacoConfig c, const __nv_bfloat16* local,
                                      __nv_bfloat16* output) {
  __shared__ int ready;
  const int fragment = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if (threadIdx.x == 0)
    ready = !taco_op3_stopped(c) && taco_op3_is_ready(c, fragment);
  __syncthreads();  // CTA acquire visibility for all participating payload readers.
  if (!ready) return;
  for (int compact = warp; compact < 16; compact += 4) {
    const int group = taco_op3_group(fragment, compact);
    int stopped = 0, done = 0;
    if (lane == 0) {
      stopped = taco_op3_stopped(c);
      done = taco_op3_done(c)[group] == c.op3_epoch;
    }
    stopped = __shfl_sync(0xffffffff, stopped, 0);
    if (stopped) return;  // Warp-uniform exit; no CTA barrier remains below.
    done = __shfl_sync(0xffffffff, done, 0);
    if (done) continue;
    if (c.op3_diagnostics && lane == 0)
      taco_op3_first(taco_op3_stats(c), taco_op3_clock());
    op3_v2_decode_warp(c, local, output, group);
    __syncwarp();  // Finish every output store before publishing this group's done.
    if (lane == 0) {
      taco_op3_done(c)[group] = c.op3_epoch;
      if (c.op3_diagnostics) {
        const auto now = taco_op3_clock();
        taco_op3_first(taco_op3_stats(c) + 3, now);
        atomicMax(taco_op3_stats(c) + 4, now);
        atomicAdd(taco_op3_stats(c) + 2, 1ULL);
        if (taco_op3_stopped(c)) atomicAdd(taco_op3_stats(c) + 6, 1ULL);
      }
    }
  }
}

// Same grouping as try: one fragment per CTA, four H128 warps. Tail can poll
// remote producers only after its OWN GEMM and finite auxiliary stream end.
__global__ void taco_op3_v2_tail_decode(FluxTacoConfig c, const __nv_bfloat16* local,
                                       __nv_bfloat16* output) {
  __shared__ int pending;
  const int fragment = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  if (threadIdx.x == 0) {
    pending = 0;
    for (int compact = 0; compact < 16; ++compact)
      pending |= taco_op3_done(c)[taco_op3_group(fragment, compact)] != c.op3_epoch;
    if (pending)
      while (!taco_op3_is_ready(c, fragment)) __nanosleep(64);
  }
  __syncthreads();
  if (!pending) return;
  for (int compact = warp; compact < 16; compact += 4) {
    const int group = taco_op3_group(fragment, compact);
    int done = lane == 0 ? taco_op3_done(c)[group] == c.op3_epoch : 0;
    done = __shfl_sync(0xffffffff, done, 0);
    if (!done) op3_v2_decode_warp(c, local, output, group);
  }
}
extern "C" int taco_op3_try(FluxTacoConfig c, const void* local, void* output,
                           cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch || !local || !output) return int(cudaErrorInvalidValue);
  for (int round = 0; round < c.op3_rounds; ++round) {
    taco_op3_v2_try_decode<<<kOp3EarlyCtas, 128, 0, stream>>>(c,
        static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
    auto error = cudaGetLastError();
    if (error != cudaSuccess) return int(error);
  }
  return int(cudaSuccess);
}
extern "C" int taco_op3_tail(FluxTacoConfig c, const void* local, void* output,
                            cudaStream_t stream) {
  if (!c.enabled || !c.op3_epoch || !local || !output) return int(cudaErrorInvalidValue);
  taco_op3_v2_tail_decode<<<kOp3Fragments, 128, 0, stream>>>(c,
      static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}
'''


def wrapper(source):
    text = legacy.wrapper(source)
    return replace_once(text, '''    cutlass_op->run(args, workspace, stream);
    if (taco.enabled && taco.op3_overlap) {''', '''    cutlass_op->run(args, workspace, stream);
    if (taco.enabled && taco.op3_overlap) {
      // Main-stream completion boundary, independent of the auxiliary stream.
      // Work already started may finish; no new row is taken after observing stop.
      CUDA_CHECK(static_cast<cudaError_t>(taco_op3_stop(taco, stream)));''')


def frontend(source):
    text = legacy.frontend(source)
    text = replace_once(text, '        self.op3_epoch = 0', '''        self.op3_epoch = 0
        if os.environ.get('H100_OP3_PROTOCOL') != 'h100-arrival-op3-v2':
            raise RuntimeError('Op3-v2 requires its own experiment protocol')''')
    text = replace_once(text, '            self.lib.taco_op3_configure.restype = ctypes.c_int',
        '''            self.lib.taco_op3_configure.restype = ctypes.c_int
            self.lib.taco_op3_protocol_version.argtypes = []
            self.lib.taco_op3_protocol_version.restype = ctypes.c_int
            if self.lib.taco_op3_protocol_version() != 2:
                raise RuntimeError('Wrong op3 library protocol')''')
    text = replace_once(text, 'start:start+18472', 'start:start+49216')
    start = text.index('    def op3_diagnostics(self):')
    end = text.index('    @property\n    def wire_bytes_per_rank(self):', start)
    return text[:start] + '''    def op3_diagnostics(self):
        if not self.op3_instrument:
            raise RuntimeError('Diagnostics are only available in untimed profile runs')
        torch.cuda.synchronize()
        start = self.rank * self.packet_bytes_per_source + 49160
        values = self.peers[self.rank][start:start+56].view(torch.int64).cpu().tolist()
        first, producer_end, count, first_end, last_end, stop, late = values
        return dict(protocol='h100-arrival-op3-v2', epoch=self.op3_epoch,
                    mode=self.op3_mode, rounds=self.op3_rounds, work_unit='h128_group',
                    try_groups=count, max_try_groups=1024, early_scan_ctas=64,
                    stop_on_local_gemm=True, stop_ns=stop, groups_finished_after_stop=late,
                    first_decode_start_ns=first, first_decode_complete_ns=first_end,
                    last_decode_complete_ns=last_end, last_local_publish_ns=producer_end,
                    useful_overlap=0 < first_end < producer_end)

''' + text[end:]


def sources(root, decoder_only=False):
    root = Path(root)
    decoder = {DECODER: (HERE/'arrival_op3_v2_decode.cuh').read_text()}
    if decoder_only:
        return {RUNTIME: decoder_only_runtime((root/RUNTIME).read_text())} | decoder
    return {path: transform((root/path).read_text()) for path, transform in (
        (HEADER, header), (RUNTIME, runtime), (EPILOGUE, epilogue),
        (WRAPPER, wrapper), (FRONTEND, frontend))} | decoder | {
            HELPER: (HERE/'arrival_op3_v2_cuda.cuh').read_text()}
