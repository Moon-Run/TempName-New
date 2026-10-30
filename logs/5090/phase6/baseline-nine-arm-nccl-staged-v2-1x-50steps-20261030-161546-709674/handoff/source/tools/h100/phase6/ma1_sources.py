"""Post-publication source prefetch; producer, barriers and ownership stay pinned."""
from pathlib import Path

from common import HERE, replace_once, require

RUNTIME = 'src/gemm_rs/taco_runtime.cu'
DECODER = 'src/gemm_rs/ma1_decode.cuh'


def sources(reference, arm):
    require(arm in ('W', 'P'), 'MA1 transforms only W/P')
    source = (Path(reference)/RUNTIME).read_text()
    source = replace_once(source, '#include "gemm_rs/taco_codec.cuh"',
                          '#include "gemm_rs/taco_codec.cuh"\n#include "gemm_rs/ma1_decode.cuh"')
    source = replace_once(source, '''  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  taco_decode_ring_kernel<<<taco_groups(c), 128, 0, stream>>>(''',
                          '''  if (!c.enabled || !local || !output || !taco_ma1_exact_shape(c))
    return int(cudaErrorInvalidValue);
  taco_ma1_decode_kernel<'''+('true' if arm == 'P' else 'false')+'''>
      <<<taco_groups(c) / 4, 128, 0, stream>>>(''')
    source += ('\nextern "C" int taco_ma1_protocol_version() { return 1; }\n'
               'extern "C" int taco_ma1_prefetch_enabled() { return '+str(int(arm == 'P'))+'; }\n')
    return {RUNTIME: source, DECODER: (HERE/'ma1_decode.cuh').read_text()}
