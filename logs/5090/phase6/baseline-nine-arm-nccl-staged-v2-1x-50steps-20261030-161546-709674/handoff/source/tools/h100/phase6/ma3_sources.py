"""Immediate source decoding: serial control D and parallel source arm S."""
from pathlib import Path

from common import HERE, replace_once, require

RUNTIME = 'src/gemm_rs/taco_runtime.cu'
DECODER = 'src/gemm_rs/ma3_decode.cuh'


def sources(reference, arm):
    require(arm in ('D', 'S'), 'MA3 transforms only D/S')
    source = (Path(reference)/RUNTIME).read_text()
    source = replace_once(source, '#include "gemm_rs/taco_codec.cuh"',
                          '#include "gemm_rs/taco_codec.cuh"\n#include "gemm_rs/ma3_decode.cuh"')
    kernel = 'parallel' if arm == 'S' else 'serial'
    blocks = 'taco_groups(c)' if arm == 'S' else 'taco_groups(c) / 4'
    source = replace_once(source, '''  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  taco_decode_ring_kernel<<<taco_groups(c), 128, 0, stream>>>(''',
                          '''  if (!c.enabled || !local || !output || !taco_ma3_exact_shape(c))
    return int(cudaErrorInvalidValue);
  taco_ma3_'''+kernel+'''_kernel<<<'''+blocks+''', 128, 0, stream>>>(''')
    source += ('\nextern "C" int taco_ma3_protocol_version() { return 1; }\n'
               'extern "C" int taco_ma3_source_parallel_enabled() { return '+str(int(arm == 'S'))+'; }\n'
               'extern "C" int taco_ma3_prefetch_enabled() { return 0; }\n')
    return {RUNTIME: source, DECODER: (HERE/'ma3_decode.cuh').read_text()}
