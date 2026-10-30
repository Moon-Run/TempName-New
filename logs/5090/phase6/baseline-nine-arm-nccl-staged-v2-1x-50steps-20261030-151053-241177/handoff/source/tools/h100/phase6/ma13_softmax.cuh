// MA13: preserve MA11's logical reduction tree while virtualizing empty warps.
// The original tree derives from PyTorch v2.6.0 SoftMax.cu/block_reduce.cuh.
// Attribution and redistribution terms: ma11_pytorch_LICENSE.txt.
// References: https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/SoftMax.cu
//             https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/block_reduce.cuh
// This header defines only the new forward ABI; backward remains original MA11.
#ifndef H100_PHASE6_MA13_SOFTMAX_CUH
#define H100_PHASE6_MA13_SOFTMAX_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>

// Keep the accurate libdevice implementation and compile with FTZ disabled.
extern "C" __device__ float __nv_expf(float);

namespace ma13_softmax {
constexpr int kColumns = 2048;
constexpr int kVector = 8;
constexpr int kThreads = kColumns / kVector;
constexpr int kRealWarps = kThreads / 32;
struct alignas(16) Vector { __nv_bfloat16 value[kVector]; };

__device__ __forceinline__ float maximum(float a, float b) {
  return a < b ? b : a;
}

template<bool IsMaximum>
__device__ __forceinline__ float shuffle_reduce(float x) {
  #pragma unroll
  for (int distance = 16; distance; distance >>= 1) {
    const float other = __shfl_down_sync(0xffffffffu, x, distance);
    x = IsMaximum ? maximum(x, other) : __fadd_rn(x, other);
  }
  return x;
}

template<bool IsMaximum>
__device__ __forceinline__ float forward_reduce(float x, float* scratch) {
  const int lane = threadIdx.x % 32;
  const int warp = threadIdx.x / 32;
  x = shuffle_reduce<IsMaximum>(x);
  __syncthreads();
  if (!lane) scratch[warp] = x;
  __syncthreads();
  if (!warp) {
    // Original logical warps 8..31 contained only neutral inputs. Their
    // complete shuffle reduction returns exactly this same neutral value.
    // Retain all 32 lanes and all original second-level shuffle operations.
    x = lane < kRealWarps ? scratch[lane] : (IsMaximum ? -FLT_MAX : 0.0f);
    x = shuffle_reduce<IsMaximum>(x);
    if (!lane) scratch[0] = x;
  }
  __syncthreads();
  return scratch[0];
}

__global__ void ma13_softmax_forward_kernel(
    const __nv_bfloat16* input, const bool* mask, __nv_bfloat16* output) {
  __shared__ float reduction[kRealWarps];
  const int vector = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * kColumns;
  const int mask_base = (blockIdx.x % kColumns) * kColumns;
  Vector x = reinterpret_cast<const Vector*>(input + base)[vector];
  float local_max = -FLT_MAX;
  #pragma unroll
  for (int i = 0; i < kVector; ++i) {
    if (mask[mask_base + vector * kVector + i])
      x.value[i] = __float2bfloat16_rn(-10000.0f);
    local_max = fmaxf(local_max, __bfloat162float(x.value[i]));
  }
  const float row_max = forward_reduce<true>(local_max, reduction);
  float exponent[kVector];
  float local_sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < kVector; ++i) {
    const float delta = __fsub_rn(__bfloat162float(x.value[i]), row_max);
    exponent[i] = __nv_expf(delta);
    local_sum = __fadd_rn(local_sum, exponent[i]);
  }
  const float denominator = forward_reduce<false>(local_sum, reduction);
  Vector y;
  #pragma unroll
  for (int i = 0; i < kVector; ++i)
    y.value[i] = __float2bfloat16_rn(__fdiv_rn(exponent[i], denominator));
  reinterpret_cast<Vector*>(output + base)[vector] = y;
}

inline bool valid(const void* input, const void* mask, const void* output,
                  int64_t rows, int64_t columns) {
  return input && mask && output && rows > 0 && rows <= 0x7fffffff &&
      rows % kColumns == 0 && columns == kColumns &&
      (reinterpret_cast<uintptr_t>(input) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(output) % 16 == 0) && input != output;
}
} // namespace ma13_softmax

extern "C" int ma13_softmax_protocol() { return 13; }

extern "C" int ma13_softmax_forward_cuda(
    const void* input, const void* mask, void* output,
    int64_t rows, int64_t columns, void* stream) {
  if (!ma13_softmax::valid(input, mask, output, rows, columns))
    return cudaErrorInvalidValue;
  ma13_softmax::ma13_softmax_forward_kernel<<<static_cast<unsigned>(rows),
      ma13_softmax::kThreads, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<const bool*>(mask),
      static_cast<__nv_bfloat16*>(output));
  return static_cast<int>(cudaPeekAtLastError());
}
#endif
