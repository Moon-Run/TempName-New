// SPDX-License-Identifier: Apache-2.0
// Independent block-absmax symmetric INT8 communication codec. This is not
// DeepSpeed kernel code or a claim of DeepSpeed bitstream compatibility.
// Wire v1: for each ascending remote peer, int8[1048576], float32[8192].
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

namespace {
constexpr int kBlock = 128;
constexpr int kElements = 256 * 4096;
constexpr int kGroups = kElements / kBlock;
constexpr size_t kPeerBytes = kElements + kGroups * sizeof(float);
inline bool aligned(const void* p, size_t n) {
  return p && reinterpret_cast<uintptr_t>(p) % n == 0;
}
inline bool rank_ok(int rank) { return rank >= 0 && rank < 8; }
__device__ __forceinline__ __nv_bfloat16 value_at(
    const unsigned char* wire, int slot, int i) {
  const unsigned char* packet = wire + size_t(slot) * kPeerBytes;
  const signed char q = reinterpret_cast<const signed char*>(packet)[i];
  const float scale = reinterpret_cast<const float*>(packet + kElements)[i / kBlock];
  return __float2bfloat16_rn(__fmul_rn(float(q), scale));
}
}  // namespace

// Each warp owns a group; logical group identity and wire location never
// depend on CTA launch/completion order or the GEMM's preceding tile order.
__global__ void oc_int8_encode_kernel(
    const __nv_bfloat16* input, unsigned char* wire, int rank) {
  const int lane = threadIdx.x & 31;
  const int group = blockIdx.x * 4 + threadIdx.x / 32;
  const int slot = blockIdx.y;
  const int destination = slot + int(slot >= rank);
  const __nv_bfloat16* values = input + size_t(destination) * kElements + group * kBlock;
  float x[4];
  float maximum = 0.f;
  bool invalid = false;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] = __bfloat162float(values[lane + j * 32]);
    invalid |= !isfinite(x[j]);
    maximum = fmaxf(maximum, fabsf(x[j]));
  }
  #pragma unroll
  for (int distance = 16; distance; distance /= 2)
    maximum = fmaxf(maximum, __shfl_xor_sync(0xffffffff, maximum, distance));
  // Propagate any invalid input through the whole block, including timed
  // paths. This does not add a synchronization or a separate GPU launch.
  invalid = __any_sync(0xffffffff, invalid);
  const float scale = invalid ? __int_as_float(0x7fc00000) :
      (maximum == 0.f ? 1.f : __fdiv_rn(maximum, 127.f));
  unsigned char* packet = wire + size_t(slot) * kPeerBytes;
  if (lane == 0) reinterpret_cast<float*>(packet + kElements)[group] = scale;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int q = invalid ? 0 : __float2int_rn(__fdiv_rn(x[j], scale));
    reinterpret_cast<signed char*>(packet)[group * kBlock + lane + j * 32] =
        static_cast<signed char>(max(-127, min(127, q)));
  }
}

__global__ void oc_int8_decode_kernel(const unsigned char* wire, __nv_bfloat16* output) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  const int slot = blockIdx.y;
  output[size_t(slot) * kElements + i] = value_at(wire, slot, i);
}

__global__ void oc_int8_decode_reduce_kernel(const unsigned char* wire,
    const __nv_bfloat16* local, __nv_bfloat16* output, int rank) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  __nv_bfloat16 accum = __float2bfloat16_rn(0.f);
  #pragma unroll
  for (int source = 0; source < 8; ++source) {
    const __nv_bfloat16 value = source == rank ? local[i] :
        value_at(wire, source - int(source > rank), i);
    // Decode BF16 rounding and each source-ordered BF16 sum are intentional.
    accum = __float2bfloat16_rn(__fadd_rn(__bfloat162float(accum), __bfloat162float(value)));
  }
  output[i] = accum;
}

extern "C" int oc_abi_version() { return 1; }
extern "C" const char* oc_error(int code) { return cudaGetErrorString(cudaError_t(code)); }
extern "C" int oc_encode(const void* input, void* wire, int rank, void* stream) {
  if (!aligned(input, 2) || !aligned(wire, 4) || !rank_ok(rank) || input == wire)
    return int(cudaErrorInvalidValue);
  oc_int8_encode_kernel<<<dim3(kGroups / 4, 7), 128, 0, (cudaStream_t)stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<unsigned char*>(wire), rank);
  return int(cudaGetLastError());
}
extern "C" int oc_decode(const void* wire, void* output, void* stream) {
  if (!aligned(wire, 4) || !aligned(output, 2) || wire == output)
    return int(cudaErrorInvalidValue);
  oc_int8_decode_kernel<<<dim3(kElements / 256, 7), 256, 0, (cudaStream_t)stream>>>(
      static_cast<const unsigned char*>(wire), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}
extern "C" int oc_decode_reduce(const void* wire, const void* local,
    void* output, int rank, void* stream) {
  if (!aligned(wire, 4) || !aligned(local, 2) || !aligned(output, 2) || !rank_ok(rank) ||
      output == local || output == wire)
    return int(cudaErrorInvalidValue);
  oc_int8_decode_reduce_kernel<<<kElements / 256, 256, 0, (cudaStream_t)stream>>>(
      static_cast<const unsigned char*>(wire), static_cast<const __nv_bfloat16*>(local),
      static_cast<__nv_bfloat16*>(output), rank);
  return int(cudaGetLastError());
}
