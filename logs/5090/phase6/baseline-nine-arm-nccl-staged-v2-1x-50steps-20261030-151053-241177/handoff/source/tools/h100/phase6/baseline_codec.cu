// Host-only adaptation of the pinned TACO kernels. No GEMM/RS fusion.
// The generated header contains the upstream device code verbatim.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <type_traits>
#include <cstdint>
#include <climits>
#define WARP_SIZE 32
#include "upstream_kernels.cuh"

static bool valid(int chunks, int64_t elements) {
  return chunks > 0 && chunks <= 65535 && elements > 0 &&
         elements <= INT_MAX && elements % 128 == 0;
}

extern "C" int baseline_taco_encode(const void* input, void* output,
                                     int chunks, int64_t elements, void* stream) {
  if (!input || !output || !valid(chunks, elements)) return int(cudaErrorInvalidValue);
  const size_t scales = size_t(elements / 128) * sizeof(float);
  fused_ash_compress_kernel<__nv_bfloat16, 128>
      <<<dim3(elements / 128, chunks), 128, 128 * sizeof(float), (cudaStream_t)stream>>>(
          input, output, nullptr, nullptr, int(elements), size_t(elements) * 2,
          size_t(elements) + 2 * scales, scales, 448.f, 1e-6f, 448.f,
          __NV_SATFINITE, __NV_E4M3);
  return int(cudaGetLastError());
}

extern "C" int baseline_taco_decode(const void* input, void* output,
                                     int chunks, int64_t elements, void* stream) {
  if (!input || !output || !valid(chunks, elements)) return int(cudaErrorInvalidValue);
  const size_t scales = size_t(elements / 128) * sizeof(float);
  fused_ash_decompress_kernel<__nv_bfloat16, 128>
      <<<dim3(elements / 128, chunks), 128, 128 * sizeof(float), (cudaStream_t)stream>>>(
          input, output, nullptr, nullptr, int(elements), size_t(elements) + 2 * scales,
          size_t(elements) * 2, scales, __NV_E4M3);
  return int(cudaGetLastError());
}

extern "C" const char* baseline_taco_error(int code) {
  return cudaGetErrorString(cudaError_t(code));
}
