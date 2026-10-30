"""A stays scalar, P is the original MA1 prefetch, V uses packed prefetch."""
from pathlib import Path

import ma1_sources
from common import HERE, replace_once, require

RUNTIME = 'src/gemm_rs/taco_runtime.cu'
DECODER = 'src/gemm_rs/ma2_decode.cuh'


def sources(reference, arm):
    require(arm in ('P', 'V'), 'MA2 transforms only P/V')
    if arm == 'P':
        return ma1_sources.sources(reference, 'P')
    source = (Path(reference)/RUNTIME).read_text()
    source = replace_once(source, '#include "gemm_rs/taco_codec.cuh"',
                          '#include "gemm_rs/taco_codec.cuh"\n#include "gemm_rs/ma2_decode.cuh"')
    source = replace_once(source, '''  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  taco_decode_ring_kernel<<<taco_groups(c), 128, 0, stream>>>(''',
                          '''  if (!c.enabled || !local || !output || !taco_ma2_exact_shape(c))
    return int(cudaErrorInvalidValue);
  taco_ma2_decode_kernel<<<taco_groups(c) / 4, 128, 0, stream>>>(''')
    source += ('\nextern "C" int taco_ma2_protocol_version() { return 1; }\n'
               'extern "C" int taco_ma2_packed_load_enabled() { return 1; }\n')
    return {RUNTIME: source, DECODER: (HERE/'ma2_decode.cuh').read_text()}
