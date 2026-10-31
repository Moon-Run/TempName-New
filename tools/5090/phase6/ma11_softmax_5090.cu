// RTX 5090 MA11 masked softmax for NGC 25.06 / PyTorch 2.8.0a0+5228986c39.
// Forward follows the fixed 2048-column persistent warp softmax reduction;
// backward follows the 128-thread vector/shared-memory block reduction.
// Upstream arithmetic/schedules: PyTorch commit 5228986c39, SoftMax.cu and
// PersistentSoftmax.cuh, BSD-3-Clause (see tools/h100/phase6/ma11_pytorch_LICENSE.txt).
// This is a fixed-shape fused implementation, not a generic ATen replacement.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

extern "C" __device__ float __nv_expf(float);

// A distinct C++ namespace prevents CUDA kernel-stub interposition with the
// historical MA11 symbols still present in the reused libflux_cuda.so.
namespace rtx5090_ma11_softmax {
constexpr int kColumns = 2048;
constexpr int kWarp = 32;
constexpr int kIterations = 64;
constexpr int kVector = 8;
struct alignas(16) Vector { __nv_bfloat16 value[kVector]; };

__global__ void ma11_softmax_forward_kernel(
    const __nv_bfloat16* input, const bool* mask, __nv_bfloat16* output, int rows) {
  const int lane = threadIdx.x;
  const int row = blockIdx.x * 4 + threadIdx.y;
  if (row >= rows) return; // Whole warps exit together.
  const int64_t base = static_cast<int64_t>(row) * kColumns;
  const int mask_base = (row % kColumns) * kColumns;
  const __nv_bfloat16 masked = __float2bfloat16_rn(-10000.0f);
  float values[kIterations];
  #pragma unroll
  for (int i = 0; i < kIterations; ++i) {
    const int column = lane + i * kWarp;
    values[i] = __bfloat162float(mask[mask_base + column] ? masked : input[base + column]);
  }
  // The unmasked ATen softmax consumes the BF16 result of masked_fill.
  float maximum = values[0];
  #pragma unroll
  for (int i = 0; i < kIterations; ++i)
    maximum = maximum > values[i] ? maximum : values[i];
  #pragma unroll
  for (int distance = 16; distance; distance >>= 1) {
    const float other = __shfl_xor_sync(0xffffffffu, maximum, distance);
    maximum = maximum < other ? other : maximum;
  }
  float sum = 0.0f;
  #pragma unroll
  for (int i = 0; i < kIterations; ++i) {
    values[i] = __nv_expf(__fsub_rn(values[i], maximum));
    sum = __fadd_rn(sum, values[i]);
  }
  #pragma unroll
  for (int distance = 16; distance; distance >>= 1)
    sum = __fadd_rn(sum, __shfl_xor_sync(0xffffffffu, sum, distance));
  #pragma unroll
  for (int i = 0; i < kIterations; ++i)
    output[base + lane + i * kWarp] = __float2bfloat16_rn(__fdiv_rn(values[i], sum));
}

__device__ __forceinline__ float block_sum(float value, float* scratch) {
  const int lane = threadIdx.x % kWarp;
  const int warp = threadIdx.x / kWarp;
  #pragma unroll
  for (int distance = 16; distance; distance >>= 1)
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, distance));
  __syncthreads();
  if (lane == 0) scratch[warp] = value;
  __syncthreads();
  value = threadIdx.x < 4 ? scratch[lane] : 0.0f;
  if (warp == 0) {
    #pragma unroll
    for (int distance = 16; distance; distance >>= 1)
      value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, distance));
  }
  if (threadIdx.x == 0) scratch[0] = value;
  __syncthreads();
  return scratch[0];
}

__global__ void ma11_softmax_backward_kernel(
    const __nv_bfloat16* gradient, const __nv_bfloat16* probability,
    const bool* mask, __nv_bfloat16* input_gradient) {
  __shared__ float reduction[4];
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
      products[part].value[i] = __float2bfloat16_rn(__fmul_rn(
          __bfloat162float(g.value[i]), __bfloat162float(probabilities[part].value[i])));
      local_sum = __fadd_rn(local_sum, __bfloat162float(products[part].value[i]));
    }
  }
  const float row_sum = block_sum(local_sum, reduction);
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
} // namespace rtx5090_ma11_softmax

extern "C" int ma11_softmax_protocol() { return 11; }
extern "C" int ma11_softmax_forward_cuda(
    const void* input, const void* mask, void* output, int64_t rows, int64_t columns, void* stream) {
  if (!rtx5090_ma11_softmax::valid(input, mask, output, rows, columns)) return cudaErrorInvalidValue;
  rtx5090_ma11_softmax::ma11_softmax_forward_kernel<<<static_cast<unsigned>((rows + 3) / 4), dim3(32, 4), 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(input),
      static_cast<const bool*>(mask), static_cast<__nv_bfloat16*>(output), static_cast<int>(rows));
  return static_cast<int>(cudaPeekAtLastError());
}
extern "C" int ma11_softmax_backward_cuda(
    const void* gradient, const void* probability, const void* mask, void* input_gradient,
    int64_t rows, int64_t columns, void* stream) {
  if (!rtx5090_ma11_softmax::valid(gradient, mask, input_gradient, rows, columns) ||
      !probability || reinterpret_cast<uintptr_t>(probability) % 16 || probability == input_gradient)
    return cudaErrorInvalidValue;
  rtx5090_ma11_softmax::ma11_softmax_backward_kernel<<<static_cast<unsigned>(rows), 128, 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(static_cast<const __nv_bfloat16*>(gradient),
      static_cast<const __nv_bfloat16*>(probability), static_cast<const bool*>(mask),
      static_cast<__nv_bfloat16*>(input_gradient));
  return static_cast<int>(cudaPeekAtLastError());
}
