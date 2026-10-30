"""MA5: reschedule CR2's exact H128 decoder and optionally its GEMM grid.

The 128x128 producer output tile, H128 wire groups, physical-tile masks, source
order, BF16 rounding, and CR2 ownership/barrier protocol remain unchanged.
``gemm_sms`` is CUTLASS's available-SM scheduling input, not an exact grid size.
"""
from pathlib import Path

from common import replace_once, require
import ma5_gemm

RUNTIME = 'src/gemm_rs/taco_runtime.cu'
DECODER = 'src/gemm_rs/ma2_c_decode.cuh'
WRAPPER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
MAPPING = 'check_mapping.cu'
DEFAULT_TUNING = dict(decoder_threads=256, decoder_blocks=528, gemm_sms=-1, gemm_tile_k=32)


def tuning_config(tuning=None):
    tuning = dict(DEFAULT_TUNING if tuning is None else tuning)
    require(set(tuning) == set(DEFAULT_TUNING), 'MA5 tuning fields changed')
    require(all(type(value) is int for value in tuning.values()),
            'MA5 tuning values must be integers')
    threads, blocks, sms, tile_k = (tuning[key] for key in DEFAULT_TUNING)
    require(threads in (64, 128, 256, 512), 'Unsupported MA5 decoder threads')
    require(1 <= blocks <= 8192 // (threads // 32),
            'MA5 decoder blocks must be positive and no greater than the full grid')
    require(sms in (-1, 128, 132), 'MA5 GEMM scheduling supports -1, 128, or 132 SMs')
    require(tile_k in (32, 64), 'MA5 GEMM tile K must be 32 or 64')
    return tuning


def decoder(source, tuning):
    """Extract the old per-warp body verbatim, then add an independent grid loop."""
    tuning = tuning_config(tuning)
    source = replace_once(source, '''__global__ void taco_ma2c_decode_kernel(const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {''',
        '''__device__ __forceinline__ void taco_ma5_decode_group(const FluxTacoConfig& c,
    const __nv_bfloat16* local, __nv_bfloat16* output, int group) {''')
    source = replace_once(source, '  const int group = blockIdx.x * 4 + threadIdx.x / 32;\n', '')
    # Keeping the BF16 early return inside this helper is essential: it must
    # finish this group's work without skipping the warp's later grid iterations.
    return source + f'''
// MA5: every warp owns one complete H128 group at a time. Loop bounds are
// warp-uniform, so the unchanged full-mask warp shuffles remain well defined.
__global__ void taco_ma5_decode_kernel(const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {{
  constexpr int kWarps = {tuning['decoder_threads'] // 32};
  for (int group = blockIdx.x * kWarps + threadIdx.x / 32;
       group < 8192; group += gridDim.x * kWarps) {{
    taco_ma5_decode_group(c, local, output, group);
  }}
}}
'''


def runtime(source, tuning):
    tuning = tuning_config(tuning)
    require('taco_cr2_protocol_version() { return 1; }' in source,
            'MA5 requires the frozen CR2 runtime')
    source = replace_once(source, 'taco_ma2c_decode_kernel<<<2048, 128, 0, stream>>>',
        f"taco_ma5_decode_kernel<<<{tuning['decoder_blocks']}, {tuning['decoder_threads']}, 0, stream>>>")
    return source + f'''
extern "C" int taco_ma5_protocol_version() {{ return 1; }}
extern "C" int taco_ma5_decoder_threads() {{ return {tuning['decoder_threads']}; }}
extern "C" int taco_ma5_decoder_blocks() {{ return {tuning['decoder_blocks']}; }}
extern "C" int taco_ma5_gemm_sms() {{ return {tuning['gemm_sms']}; }}
extern "C" int taco_ma5_gemm_tile_k() {{ return {tuning['gemm_tile_k']}; }}
'''


def wrapper(source, tuning):
    tuning = tuning_config(tuning)
    require('cr2_op->run(args, workspace, stream);' in source,
            'MA5 requires the cached CR2 submission path')
    return replace_once(source,
        '.barrier_ptrs = barrier_ptrs.data(), .avail_sms = -1,',
        f".barrier_ptrs = barrier_ptrs.data(), .avail_sms = {tuning['gemm_sms']},")


def mapping(source, tuning):
    """Audit the same tile and Stream-K scheduling input as the actual producer."""
    tuning = tuning_config(tuning)
    source = replace_once(source, '{128,128,32},1,active,',
                          f"{{128,128,{tuning['gemm_tile_k']}}},1,active,")
    source = replace_once(source, 'k%(32*tp)', f"k%({tuning['gemm_tile_k']}*tp)")
    if tuning['gemm_sms'] != -1:
        # The kernel caps the requested scheduling width at physical SM count.
        source = replace_once(source,
            'prop.multiProcessorCount,prop.multiProcessorCount,2,2,2,8);',
            f"prop.multiProcessorCount,(prop.multiProcessorCount < {tuning['gemm_sms']} ? "
            f"prop.multiProcessorCount : {tuning['gemm_sms']}),2,2,2,8);")
    return source


def sources(reference, tuning=None):
    tuning = tuning_config(tuning)
    reference = Path(reference)
    updates = {name: transform((reference/name).read_text(), tuning)
               for name, transform in ((RUNTIME, runtime), (DECODER, decoder),
                                       (WRAPPER, wrapper), (MAPPING, mapping))}
    updates.update(ma5_gemm.sources(reference, tuning['gemm_tile_k']))
    return updates
