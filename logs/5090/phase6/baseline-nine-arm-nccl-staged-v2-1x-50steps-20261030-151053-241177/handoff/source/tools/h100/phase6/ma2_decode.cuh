// MA2: aligned packets and adjacent coefficients in the MA1 source pipeline.
// H128/E4M3 arithmetic adapted from the pinned TACO decoder; attribution and
// BSD-3-Clause license: tools/a800/phase4/TACO_LICENSE.txt.
// Include after gemm_rs/taco_codec.cuh (defines FluxTacoConfig).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

inline bool taco_ma2_exact_shape(const FluxTacoConfig& c) {
  return c.enabled && c.world == 8 && c.m == 2048 && c.n == 4096 &&
         c.rank >= 0 && c.rank < 8;
}

namespace taco_ma2 {
constexpr int kGroups = 8192;
constexpr int kSourceBytes = kGroups * 136;
constexpr int kChunkElements = 256 * 4096;
constexpr int kThreads = 128;
constexpr int kBlocks = kGroups / (kThreads / 32);

// Lane l owns coefficients [4*l,4*l+3]. Four aligned FP8 coefficients occupy
// one raw register, not four byte registers. Future packets stay raw until use.
struct Packet {
  unsigned raw;
  float quant, adaptive;
};

__device__ __forceinline__ Packet load_remote(
    const FluxTacoConfig& c, int group, int source) {
  const int lane = threadIdx.x & 31;
  const unsigned char* slot = c.peers[c.rank] + source * kSourceBytes;
  const unsigned char* payload = slot + group * 128 + lane * 4;
  const float* quant = reinterpret_cast<const float*>(slot + kGroups * 128) + group;
  const float* adaptive = quant + kGroups;
  Packet p;
  // Both the group and per-lane payload offsets are multiples of four. Inline
  // memory operations retain the intended loads; no fence/wait is introduced.
  // Compiled SASS, not C++ statement order, must establish actual early issue.
  asm volatile(
      "ld.global.u32 %0, [%3];\n\t"
      "ld.global.f32 %1, [%4];\n\t"
      "ld.global.f32 %2, [%5];\n\t"
      : "=r"(p.raw), "=f"(p.quant), "=f"(p.adaptive)
      : "l"(payload), "l"(quant), "l"(adaptive)
      : "memory");
  return p;
}

__device__ __forceinline__ unsigned long long load_local(
    const FluxTacoConfig& c, const __nv_bfloat16* local, int group) {
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* address = local + c.rank * kChunkElements + group * 128 + lane * 4;
  unsigned long long raw;
  // Four BF16 values are contiguous and eight-byte aligned.
  asm volatile("ld.global.u64 %0, [%1];" : "=l"(raw) : "l"(address) : "memory");
  return raw;
}

__device__ __forceinline__ void decode(const Packet& p, float (&x)[4]) {
  const int lane = threadIdx.x & 31;
  // Each E4M3 value is exactly representable as FP16. The packed conversion is
  // elementwise identical to the scalar intrinsic, including signed zeros;
  // quant multiplication is still a separate FP32 operation per coefficient.
  const __half2 h01(__nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(p.raw), __NV_E4M3));
  const __half2 h23(__nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(p.raw >> 16), __NV_E4M3));
  // Explicit RN products are essential: adjacent d=1 register butterflies
  // otherwise let nvcc contract q*x + q*y into an FFMA, changing the original
  // rounded product. The old shuffle between multiply/add prevented that.
  x[0] = __fmul_rn(__low2float(h01), p.quant);
  x[1] = __fmul_rn(__high2float(h01), p.quant);
  x[2] = __fmul_rn(__low2float(h23), p.quant);
  x[3] = __fmul_rn(__high2float(h23), p.quant);

  // The physical coefficient still experiences distances 1,2,4,8,16,32,64
  // IN THAT ORDER. Changing lane ownership makes the first two stages local;
  // this is not the old quarter layout with its butterfly stages reordered.
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;  // d=1
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;  // d=2
  #pragma unroll
  for (int lane_distance = 1; lane_distance < 32; lane_distance *= 2) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float y = __shfl_xor_sync(0xffffffff, x[j], lane_distance);
      x[j] = (lane & lane_distance) ? y - x[j] : x[j] + y;
    }
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] *= rsqrtf(128.f);
    x[j] = (isfinite(x[j]) ? x[j] : 0.f) / fmaxf(p.adaptive, 1e-12f);
  }
}

__device__ __forceinline__ void fold(
    const Packet& packet, __nv_bfloat16 (&accum)[4]) {
  float values[4];
  decode(packet, values);
  // Preserve BOTH reference rounding points and source order. Packing the wire
  // values is independent of accumulation: no packed-add alternative is used.
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    accum[j] = __float2bfloat16(
        __bfloat162float(accum[j]) + __bfloat162float(__float2bfloat16(values[j])));
}

__device__ __forceinline__ void fold_local(
    unsigned long long raw, __nv_bfloat16 (&accum)[4]) {
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_bfloat16 value = __ushort_as_bfloat16(static_cast<unsigned short>(raw >> (16 * j)));
    accum[j] = __float2bfloat16(__bfloat162float(accum[j]) + __bfloat162float(value));
  }
}

__device__ __forceinline__ void store(
    __nv_bfloat16* output, int group, const __nv_bfloat16 (&accum)[4]) {
  const int lane = threadIdx.x & 31;
  unsigned long long raw = 0;
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    raw |= static_cast<unsigned long long>(__bfloat16_as_ushort(accum[j])) << (16 * j);
  asm volatile("st.global.u64 [%0], %1;"
               : : "l"(output + group * 128 + lane * 4), "l"(raw) : "memory");
}
}  // namespace taco_ma2

// Same invocation contract as MA1: 2048 CTAs x 128 threads, after the EXISTING
// publication barrier and before the EXISTING reuse barrier. Only the receiver
// coefficient layout/load width changes; packets, producers and source folds do
// not. There are no CTA barriers, extra streams, flags, fences or workspaces.
__global__ void taco_ma2_decode_kernel(
    const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {
  const int group = blockIdx.x * 4 + threadIdx.x / 32;
  __nv_bfloat16 accum[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) accum[j] = __float2bfloat16(0.f);
  taco_ma2::Packet current = taco_ma2::load_remote(c, group, (c.rank + 1) & 7);
  #pragma unroll 1
  for (int offset = 1; offset < 7; ++offset) {
    const taco_ma2::Packet next = taco_ma2::load_remote(c, group, (c.rank + offset + 1) & 7);
    taco_ma2::fold(current, accum);
    current = next;
  }
  const unsigned long long last = taco_ma2::load_local(c, local, group);
  taco_ma2::fold(current, accum);
  taco_ma2::fold_local(last, accum);
  taco_ma2::store(output, group, accum);
}
