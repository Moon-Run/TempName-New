"""MA5-R: retain the measured Attention tile and restore the original CR2 MLP.

The immutable local K binds the decoder function once per native instance.
No role is inferred from a mutable TLS binding during forward, and the packet
ABI, arithmetic, producer, mask, publication barrier and output ownership stay
unchanged. The original CR2 decoder is kept verbatim for the MLP path.
"""
from pathlib import Path

from common import replace_once, require
import ma5_sources
import ma5_gemm

RUNTIME = ma5_sources.RUNTIME
DECODER = ma5_sources.DECODER
WRAPPER = ma5_sources.WRAPPER
MAPPING = ma5_sources.MAPPING
HEADER = 'src/gemm_rs/taco_runtime.h'
BASELINE_TUNING = dict(decoder_threads=128, decoder_blocks=2048, gemm_sms=-1, gemm_tile_k=32)
DEFAULT_TUNING = dict(attn=dict(ma5_sources.DEFAULT_TUNING), mlp=dict(BASELINE_TUNING))


def tuning_config(tuning=None):
    tuning = DEFAULT_TUNING if tuning is None else tuning
    require(isinstance(tuning, dict) and set(tuning) == {'attn', 'mlp'}, 'MA5-R requires both roles')
    result = {role: ma5_sources.tuning_config(tuning[role]) for role in ('attn', 'mlp')}
    require(result['mlp'] == BASELINE_TUNING, 'MA5-R must preserve the original CR2 MLP')
    require(result['attn']['gemm_sms'] == -1 and result['attn']['gemm_tile_k'] == 32,
            'MA5-R isolates decoder role selection; keep the original GEMM')
    return result


def decoder(source, tuning):
    tuning = tuning_config(tuning)
    attention = ma5_sources.decoder(source, tuning['attn'])
    start = attention.index('__device__ __forceinline__ void taco_ma5_decode_group(')
    attention = attention[start:].replace('taco_ma5_decode_group', 'taco_ma5r_decode_group')
    attention = attention.replace('taco_ma5_decode_kernel', 'taco_ma5r_attention_decode_kernel')
    # Retaining the complete original kernel avoids a grid-stride approximation
    # of CR2 in the MLP control path. Its original helpers are shared verbatim.
    return source + '\n' + attention


def runtime(source, tuning):
    tuning = tuning_config(tuning)
    require('taco_cr2_protocol_version() { return 1; }' in source,
            'MA5-R requires frozen CR2 cached submission')
    attention = tuning['attn']
    result = source + f'''
extern "C" int taco_ma5r_decode_attention(FluxTacoConfig c, const void* local,
    void* output, cudaStream_t stream) {{
  if (!c.enabled || !local || !output || !c.selected || !taco_ma2_exact_shape(c))
    return int(cudaErrorInvalidValue);
  taco_ma5r_attention_decode_kernel<<<{attention['decoder_blocks']}, {attention['decoder_threads']}, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}}
extern "C" int taco_ma5r_protocol_version() {{ return 1; }}
'''
    for key in BASELINE_TUNING:
        result += f'''extern "C" int taco_ma5r_{key}(int k) {{
  return k == 512 ? {attention[key]} : k == 2048 ? {BASELINE_TUNING[key]} : -2;
}}
'''
    return result


def header(source, tuning):
    tuning_config(tuning)
    return source + '''
// MA5-R keeps the decoder selection in the immutable native instance.
extern "C" int taco_ma5r_decode_attention(FluxTacoConfig, const void*, void*, cudaStream_t);
'''


def wrapper(source, tuning):
    tuning_config(tuning)
    source = replace_once(source, '  const int cr2_k = taco_cr2_bound_k();', '''  const int cr2_k = taco_cr2_bound_k();
  // Bind once while K belongs to this instance. forward() must not consult TLS
  // for the role: Attention/MLP and double-buffered slots can be interleaved.
  const decltype(&taco_decode_reduce) ma5r_decode =
      cr2_k == 512 ? &taco_ma5r_decode_attention : &taco_decode_reduce;''')
    return replace_once(source, '''CUDA_CHECK(static_cast<cudaError_t>(taco_decode_reduce(
          owned_taco, output_buffer.data_ptr(), output.data_ptr(), stream)));''',
        '''CUDA_CHECK(static_cast<cudaError_t>(ma5r_decode(
          owned_taco, output_buffer.data_ptr(), output.data_ptr(), stream)));''')


def sources(reference, tuning=None):
    tuning = tuning_config(tuning)
    reference = Path(reference)
    return {name: transform((reference / name).read_text(), tuning)
            for name, transform in ((RUNTIME, runtime), (DECODER, decoder),
                                    (HEADER, header), (WRAPPER, wrapper))}
