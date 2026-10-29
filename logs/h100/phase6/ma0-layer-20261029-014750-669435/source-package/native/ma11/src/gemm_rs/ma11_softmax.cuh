// MA11: materialization-free mask/softmax for the frozen BF16 attention path.
// The reduction schedules follow PyTorch v2.6.0 SoftMax.cu and block_reduce.cuh.
// References: https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/SoftMax.cu
//            https://github.com/pytorch/pytorch/blob/v2.6.0/aten/src/ATen/native/cuda/block_reduce.cuh
// Attribution and redistribution terms: ma11_pytorch_LICENSE.txt.
// This is a fixed-shape implementation, not a general softmax replacement.
// Forward: the original 1024-thread/8-element/vector two-stage shuffle tree.
// Backward: the original 128-thread/8-element/vector serial-per-warp tree.
// The BF16 grad*probability intermediate is rounded before any reduction.
#ifndef H100_PHASE6_MA11_SOFTMAX_CUH
#define H100_PHASE6_MA11_SOFTMAX_CUH

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <float.h>

// Use libdevice's accurate exponential explicitly: an enclosing Flux build may
// enable --use_fast_math, which would change an ordinary expf() call.
extern "C" __device__ float __nv_expf(float);

namespace ma11_softmax {
constexpr int kColumns = 2048;
constexpr int kVector = 8;
struct alignas(16) Vector { __nv_bfloat16 value[kVector]; };

__device__ __forceinline__ float maximum(float a, float b) {
  // This matches the cross-thread Max functor (including its operand order).
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
  x = threadIdx.x < 32 ? scratch[lane] : (IsMaximum ? -FLT_MAX : 0.0f);
  if (!warp) x = shuffle_reduce<IsMaximum>(x);
  if (!threadIdx.x) scratch[0] = x;
  __syncthreads();
  return scratch[0];
}

__global__ void ma11_softmax_forward_kernel(
    const __nv_bfloat16* input, const bool* mask, __nv_bfloat16* output) {
  __shared__ Vector cached[kColumns / kVector];
  __shared__ float reduction[32];
  const int vector = threadIdx.x;
  const int64_t base = static_cast<int64_t>(blockIdx.x) * kColumns;
  const int mask_base = (blockIdx.x % kColumns) * kColumns;
  float local_max = -FLT_MAX;
  if (vector < kColumns / kVector) {
    Vector x = reinterpret_cast<const Vector*>(input + base)[vector];
    #pragma unroll
    for (int i = 0; i < kVector; ++i) {
      // masked_fill_(-10000.0) converts the finite value to BF16 first.
      if (mask[mask_base + vector * kVector + i]) x.value[i] = __float2bfloat16_rn(-10000.0f);
      local_max = fmaxf(local_max, __bfloat162float(x.value[i]));
    }
    cached[vector] = x;
  }
  const float row_max = forward_reduce<true>(local_max, reduction);
  float local_sum = 0.0f;
  if (vector < kColumns / kVector) {
    const Vector x = cached[vector];
    #pragma unroll
    for (int i = 0; i < kVector; ++i) {
      const float delta = __fsub_rn(__bfloat162float(x.value[i]), row_max);
      local_sum = __fadd_rn(local_sum, __nv_expf(delta));
    }
  }
  const float denominator = forward_reduce<false>(local_sum, reduction);
  if (vector < kColumns / kVector) {
    const Vector x = cached[vector];
    Vector y;
    #pragma unroll
    for (int i = 0; i < kVector; ++i) {
      const float delta = __fsub_rn(__bfloat162float(x.value[i]), row_max);
      y.value[i] = __float2bfloat16_rn(__fdiv_rn(__nv_expf(delta), denominator));
    }
    reinterpret_cast<Vector*>(output + base)[vector] = y;
  }
}

__device__ __forceinline__ float backward_reduce(float x, float* scratch) {
  // Deliberately NOT a shuffle/tree sum. PyTorch's backward blockReduce
  // first serially folds 32 thread sums for each warp, then four warp sums.
  __syncthreads();
  scratch[threadIdx.x] = x;
  __syncthreads();
  if (threadIdx.x < 4) {
    float warp_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 32; ++i) {
      warp_sum = __fadd_rn(warp_sum, scratch[threadIdx.x * 32 + i]);
    }
    __syncwarp(0x0fu);
    scratch[threadIdx.x] = warp_sum;
  }
  __syncthreads();
  if (!threadIdx.x) {
    float row_sum = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) row_sum = __fadd_rn(row_sum, scratch[i]);
    scratch[0] = row_sum;
  }
  __syncthreads();
  return scratch[0];
}

__global__ void ma11_softmax_backward_kernel(
    const __nv_bfloat16* gradient, const __nv_bfloat16* probability,
    const bool* mask, __nv_bfloat16* input_gradient) {
  __shared__ float reduction[128];
  const int64_t base = static_cast<int64_t>(blockIdx.x) * kColumns;
  const int mask_base = (blockIdx.x % kColumns) * kColumns;
  Vector products[2], probabilities[2];
  float local_sum = 0.0f;
  #pragma unroll
  for (int part = 0; part < 2; ++part) {
    const int vector = threadIdx.x + part * 128;
    const Vector g = reinterpret_cast<const Vector*>(gradient + base)[vector];
    probabilities[part] = reinterpret_cast<const Vector*>(probability + base)[vector];
    #pragma unroll
    for (int i = 0; i < kVector; ++i) {
      // Preserve the old separate aten::mul BF16 output rounding exactly.
      products[part].value[i] = __float2bfloat16_rn(__fmul_rn(
          __bfloat162float(g.value[i]), __bfloat162float(probabilities[part].value[i])));
      local_sum = __fadd_rn(local_sum, __bfloat162float(products[part].value[i]));
    }
  }
  const float row_sum = backward_reduce(local_sum, reduction);
  #pragma unroll
  for (int part = 0; part < 2; ++part) {
    const int vector = threadIdx.x + part * 128;
    Vector result;
    #pragma unroll
    for (int i = 0; i < kVector; ++i) {
      const float p = __bfloat162float(probabilities[part].value[i]);
      const float gp = __bfloat162float(products[part].value[i]);
      const __nv_bfloat16 dx = __float2bfloat16_rn(__fmaf_rn(-p, row_sum, gp));
      result.value[i] = mask[mask_base + vector * kVector + i] ? __float2bfloat16_rn(0.0f) : dx;
    }
    reinterpret_cast<Vector*>(input_gradient + base)[vector] = result;
  }
}

inline bool valid(const void* a, const void* mask, const void* b, int64_t rows, int64_t columns) {
  return a && mask && b && rows > 0 && rows <= 0x7fffffff &&
      rows % kColumns == 0 && columns == kColumns &&
      (reinterpret_cast<uintptr_t>(a) % 16 == 0) &&
      (reinterpret_cast<uintptr_t>(b) % 16 == 0) && a != b;
}
} // namespace ma11_softmax

extern "C" int ma11_softmax_protocol() { return 11; }

// The mask is contiguous bool[2048,2048], broadcast over head/batch rows.
// All data/output tensors are contiguous BF16[rows,2048], 16-byte aligned.
// No allocations, host copies, synchronization, or additional streams.
extern "C" int ma11_softmax_forward_cuda(
    const void* input, const void* mask, void* output,
    int64_t rows, int64_t columns, void* stream) {
  if (!ma11_softmax::valid(input, mask, output, rows, columns)) return cudaErrorInvalidValue;
  ma11_softmax::ma11_softmax_forward_kernel<<<static_cast<unsigned>(rows), 1024, 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(input),
      static_cast<const bool*>(mask), static_cast<__nv_bfloat16*>(output));
  return static_cast<int>(cudaPeekAtLastError());
}

extern "C" int ma11_softmax_backward_cuda(
    const void* gradient, const void* probability, const void* mask, void* input_gradient,
    int64_t rows, int64_t columns, void* stream) {
  if (!ma11_softmax::valid(gradient, mask, input_gradient, rows, columns) ||
      !probability || reinterpret_cast<uintptr_t>(probability) % 16 || probability == input_gradient)
    return cudaErrorInvalidValue;
  ma11_softmax::ma11_softmax_backward_kernel<<<static_cast<unsigned>(rows), 128, 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(gradient),
      static_cast<const __nv_bfloat16*>(probability), static_cast<const bool*>(mask),
      static_cast<__nv_bfloat16*>(input_gradient));
  return static_cast<int>(cudaPeekAtLastError());
}
#endif
