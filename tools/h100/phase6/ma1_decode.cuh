// MA1: source-dimension load/compute pipelining after the publication barrier.
// H128/E4M3 arithmetic adapted from the pinned TACO decoder; attribution and
// BSD-3-Clause license: tools/a800/phase4/TACO_LICENSE.txt.
// Include after gemm_rs/taco_codec.cuh (which defines FluxTacoConfig).
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// Deliberately exact-shape: no partial warps, tail coefficients, selective
// packets, or communication/publication protocol are introduced by this file.
inline bool taco_ma1_exact_shape(const FluxTacoConfig& c) {
  return c.enabled && c.world == 8 && c.m == 2048 && c.n == 4096 &&
         c.rank >= 0 && c.rank < 8;
}

namespace taco_ma1 {
constexpr int kGroups = 8192;
constexpr int kSourceBytes = kGroups * 136;
constexpr int kChunkElements = 256 * 4096;
constexpr int kThreads = 128;
constexpr int kBlocks = kGroups / (kThreads / 32);

// Keep FP8 in raw registers until its own decode begins. Converting the future
// source here would introduce a dependency that waits for the prefetched load.
struct Packet {
  unsigned p0, p1, p2, p3;
  float quant, adaptive;
};
struct Local {
  unsigned p0, p1, p2, p3;
};

// Both W and P use exactly these loads and addresses. The volatile inline PTX
// prevents nvcc from deleting/CSE-ing the future loads or expressing them only
// at their later use. A memory clobber orders them against other inline memory
// operations; it is NOT a GPU fence or a wait for the returned registers.
// ptxas may still schedule independent instructions. The SM90 validation must
// inspect SASS: future-source LDG must precede current-source SHFL arithmetic.
__device__ __forceinline__ Packet load_remote(
    const FluxTacoConfig& c, int group, int source) {
  const int lane = threadIdx.x & 31;
  const unsigned char* slot = c.peers[c.rank] + source * kSourceBytes;
  const unsigned char* payload = slot + group * 128 + lane;
  const float* quant = reinterpret_cast<const float*>(slot + kGroups * 128) + group;
  const float* adaptive = quant + kGroups;
  Packet p;
  asm volatile(
      "ld.global.u8 %0, [%6];\n\t"
      "ld.global.u8 %1, [%6+32];\n\t"
      "ld.global.u8 %2, [%6+64];\n\t"
      "ld.global.u8 %3, [%6+96];\n\t"
      "ld.global.f32 %4, [%7];\n\t"
      "ld.global.f32 %5, [%8];\n\t"
      : "=r"(p.p0), "=r"(p.p1), "=r"(p.p2), "=r"(p.p3),
        "=f"(p.quant), "=f"(p.adaptive)
      : "l"(payload), "l"(quant), "l"(adaptive)
      : "memory");
  return p;
}

__device__ __forceinline__ Local load_local(
    const FluxTacoConfig& c, const __nv_bfloat16* local, int group) {
  const int lane = threadIdx.x & 31;
  const __nv_bfloat16* address = local + c.rank * kChunkElements + group * 128 + lane;
  Local p;
  asm volatile(
      "ld.global.u16 %0, [%4];\n\t"
      "ld.global.u16 %1, [%4+64];\n\t"
      "ld.global.u16 %2, [%4+128];\n\t"
      "ld.global.u16 %3, [%4+192];\n\t"
      : "=r"(p.p0), "=r"(p.p1), "=r"(p.p2), "=r"(p.p3)
      : "l"(address)
      : "memory");
  return p;
}

__device__ __forceinline__ float fp8(unsigned raw) {
  const __half_raw h = __nv_cvt_fp8_to_halfraw(static_cast<unsigned char>(raw), __NV_E4M3);
  return __half2float(__half(h));
}

// Four independent 32-coefficient chains share one warp. W and P interleave
// these chains identically, so the P/W comparison isolates source prefetch.
// For EACH coefficient this is the pinned scalar sequence: FP8 -> FP32 * q,
// distances 1,2,4,8,16,32,64, rsqrt(128), finite filter, then / max(a,1e-12).
// No scale reassociation, shared inverse transform, or FP32 source sum occurs.
__device__ __forceinline__ void decode(const Packet& p, float (&x)[4]) {
  const int lane = threadIdx.x & 31;
  x[0] = fp8(p.p0) * p.quant;
  x[1] = fp8(p.p1) * p.quant;
  x[2] = fp8(p.p2) * p.quant;
  x[3] = fp8(p.p3) * p.quant;
  #pragma unroll
  for (int distance = 1; distance < 32; distance *= 2) {
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float y = __shfl_xor_sync(0xffffffff, x[j], distance);
      x[j] = (lane & distance) ? y - x[j] : x[j] + y;
    }
  }
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] *= rsqrtf(128.f);
    x[j] = (isfinite(x[j]) ? x[j] : 0.f) / fmaxf(p.adaptive, 1e-12f);
  }
}

__device__ __forceinline__ void fold(
    const Packet& p, __nv_bfloat16 (&accum)[4]) {
  float values[4];
  decode(p, values);
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    accum[j] = __float2bfloat16(
        __bfloat162float(accum[j]) + __bfloat162float(__float2bfloat16(values[j])));
}

__device__ __forceinline__ void fold_local(
    const Local& p, __nv_bfloat16 (&accum)[4]) {
  const unsigned raw[4] = {p.p0, p.p1, p.p2, p.p3};
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __nv_bfloat16 value = __ushort_as_bfloat16(static_cast<unsigned short>(raw[j]));
    accum[j] = __float2bfloat16(__bfloat162float(accum[j]) + __bfloat162float(value));
  }
}
}  // namespace taco_ma1

// Called only after the EXISTING publication barrier, with the existing reuse
// barrier still following this kernel. Exactly 2048 CTAs x 128 threads.
// Prefetch=false (W): load source s -> decode/fold source s.
// Prefetch=true  (P): load s+1 -> decode/fold s -> rotate register packets.
// There are no readiness flags, auxiliary streams, asynchronous copies, atomics,
// fences, CTA barriers, or changes to the producer/packet/output ownership.
template <bool Prefetch>
__global__ void taco_ma1_decode_kernel(
    const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {
  // SM90/CUDA12.8: keep the dynamically indexed peer array in kernel parameter
  // storage. Without grid_constant nvcc materializes an 88-byte local copy of
  // the by-value configuration per thread. Both W and P use this annotation.
  const int group = blockIdx.x * 4 + threadIdx.x / 32;
  const int lane = threadIdx.x & 31;
  __nv_bfloat16 accum[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) accum[j] = __float2bfloat16(0.f);

  if constexpr (Prefetch) {
    taco_ma1::Packet current = taco_ma1::load_remote(c, group, (c.rank + 1) & 7);
    // Six rotations keep only current/next live, rather than all seven packets.
    // Disable source unrolling so register lifetime and code size stay bounded.
    #pragma unroll 1
    for (int offset = 1; offset < 7; ++offset) {
      const taco_ma1::Packet next = taco_ma1::load_remote(c, group, (c.rank + offset + 1) & 7);
      taco_ma1::fold(current, accum);
      current = next;
    }
    // The ring's last source is local BF16. Prefetch that data while the seventh
    // remote source is decoded; never read the reserved local FP8 packet slot.
    const taco_ma1::Local last = taco_ma1::load_local(c, local, group);
    taco_ma1::fold(current, accum);
    taco_ma1::fold_local(last, accum);
  } else {
    #pragma unroll 1
    for (int offset = 1; offset < 8; ++offset) {
      const taco_ma1::Packet current = taco_ma1::load_remote(c, group, (c.rank + offset) & 7);
      taco_ma1::fold(current, accum);
    }
    const taco_ma1::Local last = taco_ma1::load_local(c, local, group);
    taco_ma1::fold_local(last, accum);
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    output[group * 128 + lane + j * 32] = accum[j];
}
