"""Only replace frozen C's receiver; its masks, producers and lifecycle stay C."""
from pathlib import Path
from common import HERE, replace_once

RUNTIME = 'src/gemm_rs/taco_runtime.cu'
DECODER = 'src/gemm_rs/ma2_c_decode.cuh'


def sources(reference):
    source = (Path(reference)/RUNTIME).read_text()
    source = replace_once(source, '#include "gemm_rs/taco_codec.cuh"',
                          '#include "gemm_rs/taco_codec.cuh"\n#include "gemm_rs/ma2_c_decode.cuh"')
    source = replace_once(source, '''  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  int variant = decode_variant < 0 ?''', '''  if (!c.enabled || !local || !output || !c.selected || !taco_ma2_exact_shape(c))
    return int(cudaErrorInvalidValue);
  taco_ma2c_decode_kernel<<<2048, 128, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
  int variant = decode_variant < 0 ?''')
    # Keep old kernels available for static comparison, but remove unreachable
    # dispatch rather than exposing an alternative fallback to measurement.
    begin = source.index('  int variant = decode_variant < 0 ?')
    source = source[:begin] + '}\n'
    source += '\nextern "C" int taco_ma2c_protocol_version() { return 1; }\n'
    return {RUNTIME: source, DECODER: (HERE/'ma2_c_decode.cuh').read_text(),
            'src/gemm_rs/ma2_decode.cuh': (HERE/'ma2_decode.cuh').read_text()}
