"""MA12: exact MA89 producer/consumer arithmetic with fragment publication.

Input is an already registered MA89 F4-cache-zcount source tree. The control is
not transformed. Completion epochs guard reuse of each existing private slot;
fragment epochs replace the current-round all-GPU publication rendezvous.
"""
from pathlib import Path
from common import HERE, replace_once, require
import ma9_sources as zero_count

RUNTIME, HEADER = zero_count.RUNTIME, zero_count.HEADER
WRAPPER, DECODER = zero_count.WRAPPER, zero_count.DECODER
ZERO_COUNT_HEADER = zero_count.ZERO_COUNT_HEADER
ZERO_COUNT_INCLUDE = zero_count.ZERO_COUNT_INCLUDE
ZERO_COUNT_EXPORTS = zero_count.NATIVE_EXPORTS
VERSION_SCRIPT = zero_count.VERSION_SCRIPT
EPILOGUE = 'src/gemm_rs/epilogue_evt.hpp'
READY_HEADER = 'src/gemm_rs/ma12_tile_ready.cuh'
NATIVE_EXPORTS = ('taco_ma12_protocol_version', 'taco_ma12_wait_reuse',
                  'taco_ma12_publish_done', 'taco_ma12_decode_reduce')
VARIANTS = ('F4-cache-zcount', 'F4-cache-zcount-tilepub')
BASELINE_TUNING = zero_count.BASELINE_TUNING
ROLES, ROLE_K = zero_count.ROLES, zero_count.ROLE_K
DEFAULT_TUNING = zero_count.DEFAULT_TUNING
tuning_config = zero_count.tuning_config


def kernel_name(role, tuning=None, arm=VARIANTS[0]):
    tuning_config(tuning)
    require(role in ROLES and arm in VARIANTS, 'Unknown MA12 role or arm')
    return ('taco_ma4f_decode_consumer_kernel' if arm == VARIANTS[0]
            else 'taco_ma12_decode_consumer_kernel')


def header(source):
    source = replace_once(source, '  int deferred_reuse_barrier = 0;',
        '  int deferred_reuse_barrier = 0;\n'
        '  // MA12: independent IPC allocation owned by each existing slot.\n'
        '  int* ma12_ready[8]{};\n'
        '  int ma12_epoch = 0;')
    return source + '''
extern "C" int taco_ma12_protocol_version();
extern "C" int taco_ma12_wait_reuse(FluxTacoConfig, cudaStream_t);
extern "C" int taco_ma12_publish_done(FluxTacoConfig, cudaStream_t);
extern "C" int taco_ma12_decode_reduce(FluxTacoConfig, const void*, void*,
    const void*, const void*, cudaStream_t);
'''


def epilogue(source):
    source = replace_once(source, '#include "gemm_rs/taco_codec.cuh"',
        '#include "gemm_rs/taco_codec.cuh"\n#include "gemm_rs/ma12_tile_ready.cuh"')
    source = replace_once(source, '          taco_end_step(step_idx);',
        '          taco_end_step(step_idx);\n'
        '          taco_ma12::publish_fragment(params.taco, dst_rank, tile_idx, step_idx);')
    # Only the BF16-store branch reaches this location; the encoded branch
    # returns after its identical encoder and the publication added above.
    anchor = '''      }
    }

#ifdef FLUX_TACO_BASELINE
#ifdef FLUX_TACO_SEPARATE
    CUTLASS_DEVICE void
    taco_stage_step'''
    source = replace_once(source, anchor, '''      }
#ifdef FLUX_TACO_BASELINE
      if constexpr (kTacoSupported) {
        if (params.taco.enabled)
          taco_ma12::publish_fragment(params.taco, dst_rank, tile_idx, step_idx);
      }
#endif
    }

#ifdef FLUX_TACO_BASELINE
#ifdef FLUX_TACO_SEPARATE
    CUTLASS_DEVICE void
    taco_stage_step''')
    return source


def decoder(source):
    source = replace_once(source, '#include "gemm_rs/ma2_c_decode.cuh"',
        '#include "gemm_rs/ma2_c_decode.cuh"\n#include "gemm_rs/ma12_tile_ready.cuh"')
    # Keep the original MA4 decoder for the explicit projection reference ABI;
    # the active consumer kernel has its own unambiguous name.
    start = source.index('__global__ void taco_ma4f_decode_consumer_kernel(')
    candidate = source[start:].replace('taco_ma4f_decode_consumer_kernel',
                                      'taco_ma12_decode_consumer_kernel', 1)
    candidate = replace_once(candidate,
        '      uint2 value = *reinterpret_cast<const uint2*>(local + src * taco_ma2::kChunkElements + offset);',
        '      taco_ma12::wait_source(c, group, src);\n'
        '      uint2 value = *reinterpret_cast<const uint2*>(local + src * taco_ma2::kChunkElements + offset);')
    load = 'taco_ma2c::load(c, local, group, src, selected[src])'
    require(candidate.count(load) == 2, 'MA12 expected the two original pipelined packet loads')
    candidate = candidate.replace(load,
                                  'taco_ma12_load(c, local, group, src, selected[src])')
    helper = '''
__device__ __forceinline__ taco_ma2c::Packet taco_ma12_load(
    const FluxTacoConfig& c, const __nv_bfloat16* local,
    int group, int source, bool quantized) {
  taco_ma12::wait_source(c, group, source);
  return taco_ma2c::load(c, local, group, source, quantized);
}
'''
    return source + helper + '\n' + candidate


RUNTIME_ADDITION = r'''

// A producer waits only for the previous use of its private double-buffer
// slot. There is no current-epoch all-rank publication barrier.
__global__ void taco_ma12_wait_reuse_kernel(const __grid_constant__ FluxTacoConfig c) {
  const int receiver = threadIdx.x;
  if (receiver < 8) {
    while (taco_ma12::acquire(c.ma12_ready[receiver] + taco_ma12::kReadyWords)
           < c.ma12_epoch - 1) __nanosleep(64);
  }
}

// A separate same-stream kernel is the grid-wide completion dependency for
// all 2048 decoder CTAs. No unsafe last-CTA counter or in-kernel grid barrier.
__global__ void taco_ma12_publish_done_kernel(const __grid_constant__ FluxTacoConfig c) {
  if (threadIdx.x == 0)
    taco_ma12::release(c.ma12_ready[c.rank] + taco_ma12::kReadyWords, c.ma12_epoch);
}

static bool taco_ma12_valid(const FluxTacoConfig& c) {
  if (!c.enabled || !c.selected || !taco_ma2_exact_shape(c) ||
      !c.deferred_reuse_barrier || c.allow_capture || c.ma12_epoch <= 0) return false;
  for (int i = 0; i < 8; ++i) if (!c.ma12_ready[i]) return false;
  return true;
}
extern "C" int taco_ma12_protocol_version() { return 1; }
extern "C" int taco_ma12_wait_reuse(FluxTacoConfig c, cudaStream_t stream) {
  if (!taco_ma12_valid(c)) return int(cudaErrorInvalidValue);
  taco_ma12_wait_reuse_kernel<<<1, 32, 0, stream>>>(c);
  return int(cudaGetLastError());
}
extern "C" int taco_ma12_publish_done(FluxTacoConfig c, cudaStream_t stream) {
  if (!taco_ma12_valid(c)) return int(cudaErrorInvalidValue);
  taco_ma12_publish_done_kernel<<<1, 32, 0, stream>>>(c);
  return int(cudaGetLastError());
}
extern "C" int taco_ma12_decode_reduce(FluxTacoConfig c, const void* local,
    void* output, const void* bias, const void* residual, cudaStream_t stream) {
  if (!taco_ma12_valid(c) || !local || !output || !bias || !residual)
    return int(cudaErrorInvalidValue);
  taco_ma12_decode_consumer_kernel<<<2048, 128, 0, stream>>>(c,
      static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output),
      static_cast<const __nv_bfloat16*>(bias), static_cast<const __nv_bfloat16*>(residual));
  return int(cudaGetLastError());
}
'''


def wrapper(source):
    source = replace_once(source, '  bool cr2_failed = false;', '''  bool cr2_failed = false;
  std::vector<torch::Tensor> ma12_ready_buffers;
  int ma12_epoch = 0;''')
    source = replace_once(source,
        '''      // Input and optimizer-updated weight STORAGE may change between calls.''',
        '''      // Allocate once per existing native double-buffer slot. This is
      // initialization, never a per-call workspace clear or allocation.
      if (ma12_ready_buffers.empty()) {
        ma12_ready_buffers = flux_create_tensor_list({4097}, c10::ScalarType::Int, this->group_.get());
        CUDA_CHECK(cudaMemsetAsync(ma12_ready_buffers[rank].data_ptr(), 0,
                                   4097 * sizeof(int), stream));
        group_barrier.barrier_all(stream);
      }
      TORCH_CHECK(ma12_epoch < 0x7ffffffe, "MA12 epoch exhaustion; reconstruct the experiment");
      FluxTacoConfig ma12_taco = owned_taco;
      ma12_taco.ma12_epoch = ++ma12_epoch;
      for (int i = 0; i < 8; ++i)
        ma12_taco.ma12_ready[i] = static_cast<int*>(ma12_ready_buffers[i].data_ptr());
      // CUTLASS run() rebuilds Params from the current TLS configuration.
      taco_set_config(ma12_taco);
      CUDA_CHECK(static_cast<cudaError_t>(taco_ma12_wait_reuse(ma12_taco, stream)));
      // Input and optimizer-updated weight STORAGE may change between calls.''')
    source = replace_once(source, '''      // Identical C publication rendezvous. Double buffering continues to guard
      // previous consumers before a later invocation reuses this source slot.
      group_barrier.barrier_all(stream);''',
        '''      // Explicit projection-only reference retains its old rendezvous.
      // The measured consumer acquires exactly the fragment it will load.
      if (!ma4_consumer_active) group_barrier.barrier_all(stream);''')
    source = replace_once(source, 'taco_ma4f_decode_reduce(owned_taco,',
                          'taco_ma12_decode_reduce(ma12_taco,')
    source = replace_once(source, '''      ++cr2_calls;
      return output;''', '''      CUDA_CHECK(static_cast<cudaError_t>(taco_ma12_publish_done(ma12_taco, stream)));
      ++cr2_calls;
      return output;''')
    return source


def sources(reference):
    reference = Path(reference)
    runtime = (reference / RUNTIME).read_text()
    require(ZERO_COUNT_INCLUDE in runtime and 'taco_ma12' not in runtime,
            'MA12 requires a fresh original MA89 native tree')
    result = {HEADER: header((reference / HEADER).read_text()),
              EPILOGUE: epilogue((reference / EPILOGUE).read_text()),
              DECODER: decoder((reference / DECODER).read_text()),
              WRAPPER: wrapper((reference / WRAPPER).read_text()),
              READY_HEADER: (HERE / 'ma12_tile_ready.cuh').read_text(),
              RUNTIME: runtime + RUNTIME_ADDITION}
    result[VERSION_SCRIPT] = replace_once((reference / VERSION_SCRIPT).read_text(), 'global:\n',
        'global:\n' + ''.join('    ' + name + ';\n' for name in NATIVE_EXPORTS))
    return result
