// CR2 decoder plus post-reduction bias/residual; preserve two BF16 roundings.
// TACO attribution: tools/a800/phase4/TACO_LICENSE.txt.
#pragma once
#include "gemm_rs/ma2_c_decode.cuh"

// Match aten BF16 x+bias, then residual+out: two distinct rounded additions.
// The original ordered reduction is already complete before this function.
__device__ __forceinline__ void taco_ma4f_store(
    __nv_bfloat16* output, const __nv_bfloat16* bias,
    const __nv_bfloat16* residual, int64_t offset, uint2 packed) {
  const uint2 b = *reinterpret_cast<const uint2*>(bias + offset % 4096);
  const uint2 r = *reinterpret_cast<const uint2*>(residual + offset);
  auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
  const auto* add_bias = reinterpret_cast<const __nv_bfloat162*>(&b);
  const auto* add_residual = reinterpret_cast<const __nv_bfloat162*>(&r);
  sum[0] = __hadd2(sum[0], add_bias[0]);
  sum[1] = __hadd2(sum[1], add_bias[1]);
  sum[0] = __hadd2(add_residual[0], sum[0]);
  sum[1] = __hadd2(add_residual[1], sum[1]);
  *reinterpret_cast<uint2*>(output + offset) = packed;
}

__global__ void taco_ma4f_decode_consumer_kernel(const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output,
    const __nv_bfloat16* bias, const __nv_bfloat16* residual) {
  const int lane = threadIdx.x & 31;
  const int group = blockIdx.x * 4 + threadIdx.x / 32;
  const int row = group / 32;
  const int physical_tile = (c.rank * 2 + row / 128) * 32 + group % 32;
  bool selected[8];
  bool any_quant = false;
  #pragma unroll
  for (int src = 0; src < 8; ++src) {
    selected[src] = src != c.rank && c.selected[src * 512 + physical_tile];
    any_quant |= selected[src];
  }
  if (!any_quant) {
    // Preserve C's existing fast BF16 path, including its packed BF16 adds.
    const int64_t offset = int64_t(group) * 128 + lane * 4;
    uint2 packed = make_uint2(0, 0);
    auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
    #pragma unroll
    for (int i = 1; i <= 8; ++i) {
      const int src = (c.rank + i) & 7;
      uint2 value = *reinterpret_cast<const uint2*>(local + src * taco_ma2::kChunkElements + offset);
      auto* x = reinterpret_cast<__nv_bfloat162*>(&value);
      sum[0] = __hadd2(sum[0], x[0]);
      sum[1] = __hadd2(sum[1], x[1]);
    }
    taco_ma4f_store(output, bias, residual, offset, packed);
    return;
  }
  __nv_bfloat16 accum[4];
  #pragma unroll
  for (int j = 0; j < 4; ++j) accum[j] = __float2bfloat16(0.f);
  int src = (c.rank + 1) & 7;
  auto current = taco_ma2c::load(c, local, group, src, selected[src]);
  #pragma unroll 1
  for (int offset = 1; offset < 8; ++offset) {
    src = (c.rank + offset + 1) & 7;
    const auto next = taco_ma2c::load(c, local, group, src, selected[src]);
    taco_ma2c::fold(current, accum);
    current = next;
  }
  taco_ma2c::fold(current, accum);
  uint2 packed;
  auto* pair = reinterpret_cast<__nv_bfloat162*>(&packed);
  pair[0] = __halves2bfloat162(accum[0], accum[1]);
  pair[1] = __halves2bfloat162(accum[2], accum[3]);
  taco_ma4f_store(output, bias, residual, int64_t(group) * 128 + lane * 4, packed);
}
