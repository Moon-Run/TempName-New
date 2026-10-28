"""Producer-only fusion derived from the unchanged frozen fused-v2 baseline."""
from pathlib import Path

from common import HERE, replace_once, require

RUNTIME = 'src/gemm_rs/taco_runtime.cu'
HEADER = 'src/gemm_rs/taco_runtime.h'
WRAPPER = 'src/gemm_rs/ths_op/gemm_reduce_scatter.cc'
DECODER = 'src/gemm_rs/a_baseline_decode.cuh'


def sources(reference):
    reference = Path(reference)
    runtime = (reference/RUNTIME).read_text()
    start = runtime.index('__global__ void taco_decode_ring_kernel(')
    end = runtime.index('\nextern "C" int taco_decode_reduce(', start)
    # Do not retain an unused old decoder symbol that could hide a fallback.
    runtime = runtime[:start] + '#include "gemm_rs/a_baseline_decode.cuh"\n' + runtime[end:]
    old = '''extern "C" int taco_decode_reduce(FluxTacoConfig c, const void* local, void* output,
                                   cudaStream_t stream) {
  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  taco_decode_ring_kernel<<<taco_groups(c), 128, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}'''
    new = '''extern "C" int taco_basic_decode_reduce(FluxTacoConfig c, const void* local,
                                         void* decoded, void* output,
                                         cudaStream_t stream) {
  if (!c.enabled || !local || !decoded || !output || c.world != 8 ||
      c.rank < 0 || c.rank >= c.world || c.m != 2048 || c.n != 4096)
    return int(cudaErrorInvalidValue);
  taco_basic_decode_kernel<<<dim3(taco_groups(c), c.world - 1), 128, 0, stream>>>(
      c, static_cast<__nv_bfloat16*>(decoded));
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return int(status);
  taco_basic_reduce_kernel<<<taco_groups(c), 128, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(local),
      static_cast<const __nv_bfloat16*>(decoded), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}'''
    runtime = replace_once(runtime, old, new)
    header = replace_once((reference/HEADER).read_text(),
        '''extern "C" int taco_decode_reduce(FluxTacoConfig config, const void* local_bf16,
                                   void* output_bf16, cudaStream_t stream);''',
        '''extern "C" int taco_basic_decode_reduce(FluxTacoConfig config, const void* local_bf16,
                                         void* decoded_bf16, void* output_bf16,
                                         cudaStream_t stream);''')
    wrapper = replace_once((reference/WRAPPER).read_text(),
        '''      auto output = torch::empty({taco.m / taco.world, taco.n}, buffer.options());
      auto stream = c10::cuda::getCurrentCUDAStream();
      CUDA_CHECK(static_cast<cudaError_t>(taco_decode_reduce(
          taco, buffer.data_ptr(), output.data_ptr(), stream)));''',
        '''      auto output = torch::empty({taco.m / taco.world, taco.n}, buffer.options());
      // This invocation owns its intermediate BF16 matrix. Allocation and both
      // consumers use the current stream; the caching allocator tracks lifetime.
      // It never aliases returned outputs or another invocation's peer packets.
      auto decoded = torch::empty({taco.world - 1, taco.m / taco.world, taco.n},
                                  buffer.options());
      auto stream = c10::cuda::getCurrentCUDAStream();
      CUDA_CHECK(static_cast<cudaError_t>(taco_basic_decode_reduce(
          taco, buffer.data_ptr(), decoded.data_ptr(), output.data_ptr(), stream)));''')
    require('taco_decode_ring_kernel' not in runtime, 'Old decoder was not removed')
    return {RUNTIME: runtime, HEADER: header, WRAPPER: wrapper,
            DECODER: (HERE/'a_baseline_decode.cuh').read_text()}
