// Warp decoder for the pinned H128/E4M3 all-remote A packet layout.
// Adapted from the scalar codec; attribution/license:
// tools/a800/phase4/TACO_LICENSE.txt (BSD-3-Clause).
// Include after FluxTacoConfig, taco_groups() and taco_source_bytes().
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// All 32 lanes participate. Lane l owns scalar-CTA indices l+32*j, j=0..3.
// The expression order matches flux_taco::decode() in fused_v2 exactly:
// E4M3 -> float -> qs multiply -> H128 -> normalization -> finite filter -> as
// divide. No encode changes or alternate normalization are introduced here.
__device__ __forceinline__ void op3_v2_decode_values_warp(
    const unsigned char* payload, float qs, float as, float (&x)[4]) {
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    __half_raw raw = __nv_cvt_fp8_to_halfraw(payload[lane + 32 * j], __NV_E4M3);
    x[j] = __half2float(__half(raw)) * qs;
    #pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      const float y = __shfl_xor_sync(0xffffffff, x[j], d);
      x[j] = (lane & d) ? y - x[j] : x[j] + y;
    }
  }
  // Scalar d=32 combines quarters (0,1) and (2,3), then scalar d=64
  // combines (0,2) and (1,3). Snapshot every stage before overwriting it.
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] *= rsqrtf(128.f);
    x[j] = (isfinite(x[j]) ? x[j] : 0.f) / fmaxf(as, 1e-12f);
  }
}

// One warp owns an entire group. A 128-thread CTA may invoke this for four
// independent groups; there are no CTA barriers or shared scratch to serialize
// the warps. The caller owns readiness/acquire, group bounds and output
// ownership; every lane must enter together even for padded N coefficients.
__device__ __forceinline__ void op3_v2_decode_warp(
    const FluxTacoConfig& c, const __nv_bfloat16* local, __nv_bfloat16* output,
    int64_t group) {
  const int lane = threadIdx.x & 31;
  const int nt = (c.n + 127) / 128;
  const int row = group / nt, column = (group % nt) * 128 + lane;
  const int64_t groups = taco_groups(c), source_bytes = taco_source_bytes(c);
  const int64_t chunk = int64_t(c.m / c.world) * c.n;
  __nv_bfloat16 accum[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) accum[j] = __float2bfloat16(0.f);
  // Preserve the scalar reference's source order and BOTH BF16 rounding points.
  // In particular, accumulating decoded FP32 sources and rounding just once
  // would change training arithmetic even though the H128 transform is equal.
  for (int i = 1; i <= c.world; ++i) {
    const int src = (c.rank + i) % c.world;
    float value[4];
    if (src == c.rank) {
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const int col = column + 32 * j;
        value[j] = col < c.n
            ? __bfloat162float(local[src * chunk + int64_t(row) * c.n + col])
            : 0.f;
      }
    } else {
      const unsigned char* slot = c.peers[c.rank] + src * source_bytes;
      const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
      const float* as = qs + groups;
      op3_v2_decode_values_warp(slot + group * 128, qs[group], as[group], value);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      accum[j] = __float2bfloat16(
          __bfloat162float(accum[j]) + __bfloat162float(__float2bfloat16(value[j])));
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int col = column + 32 * j;
    if (col < c.n) output[int64_t(row) * c.n + col] = accum[j];
  }
}
