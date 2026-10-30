// TACO/COCCL adaptation; see tools/a800/phase4/TACO_LICENSE.txt.
// Reduction-closure experiment: identical warp ownership in every arm.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>

// H128 is shared across sources only in the M arms. Its normalization and
// butterflies match the encoder's orthogonal transform, with FP32 arithmetic.
__device__ __forceinline__ void taco_closure_inverse(float (&x)[4]) {
  const int lane = threadIdx.x & 31;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    #pragma unroll
    for (int distance = 1; distance < 32; distance *= 2) {
      const float partner = __shfl_xor_sync(0xffffffff, x[j], distance);
      x[j] = (lane & distance) ? partner - x[j] : x[j] + partner;
    }
  }
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
  #pragma unroll
  for (int j = 0; j < 4; ++j) x[j] *= rsqrtf(128.f);
}

// All lanes enter together. A warp owns one output row/group; each lane owns
// four coefficients. A CTA has four independent warps in SP/CP/SM/CM alike.
// Finite valid packet data is required. Bench validation rejects nonfinite
// payloads/scales/output rather than silently accepting altered sanitization.
__device__ __forceinline__ void taco_closure_decode_warp(
    const FluxTacoConfig& c, const __nv_bfloat16* local, __nv_bfloat16* output,
    int64_t group) {
  const int lane = threadIdx.x & 31;
  const int nt = c.n / 128;
  const int row = group / nt, tile_n = group % nt;
  const int physical_tile = c.rank * 64 + (row / 128) * nt + tile_n;
  const int64_t groups = taco_groups(c), source_bytes = taco_source_bytes(c);
  const int64_t chunk = int64_t(c.m / c.world) * c.n;
  float compressed[4] = {}, dense[4] = {};
  bool has_compressed = false;
  for (int offset = 1; offset <= c.world; ++offset) {
    const int src = (c.rank + offset) % c.world;
    if (taco_closure_selected(c, src, c.rank, physical_tile)) {
      has_compressed = true;
      const unsigned char* slot = c.peers[c.rank] + src * source_bytes;
      const float* quant = reinterpret_cast<const float*>(slot + groups * 128);
      const float* adaptive = quant + groups;
      // P and M BOTH move scalar scaling before H. Their only arithmetic
      // difference is whether addition happens before or after the inverse.
      const float scale = quant[group] / fmaxf(adaptive[group], 1e-12f);
      float value[4];
      #pragma unroll
      for (int j = 0; j < 4; ++j) {
        const auto raw = __nv_cvt_fp8_to_halfraw(
            slot[group * 128 + lane + 32 * j], __NV_E4M3);
        value[j] = __half2float(__half(raw)) * scale;
      }
#if !FLUX_CLOSURE_MERGED
      taco_closure_inverse(value);
#endif
      #pragma unroll
      for (int j = 0; j < 4; ++j) compressed[j] += value[j];
    } else {
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        dense[j] += __bfloat162float(
            local[src * chunk + int64_t(row) * c.n + tile_n * 128 + lane + 32 * j]);
    }
  }
#if FLUX_CLOSURE_MERGED
  if (has_compressed) taco_closure_inverse(compressed);
#endif
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    output[int64_t(row) * c.n + tile_n * 128 + lane + 32 * j] =
        __float2bfloat16(compressed[j] + dense[j]);
}
