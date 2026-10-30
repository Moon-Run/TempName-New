// MA2-R receive pipeline adapted to the frozen C mixed BF16/FP8 protocol.
// TACO attribution: tools/a800/phase4/TACO_LICENSE.txt.
#pragma once
#include "gemm_rs/ma2_decode.cuh"

namespace taco_ma2c {
struct Packet {
  taco_ma2::Packet fp8;
  unsigned long long bf16;
  bool quantized;
};

__device__ __forceinline__ Packet load(const FluxTacoConfig& c,
    const __nv_bfloat16* local, int group, int source, bool quantized) {
  Packet p{};
  p.quantized = quantized;
  if (quantized) {
    p.fp8 = taco_ma2::load_remote(c, group, source);
  } else {
    const auto* address = local + source * taco_ma2::kChunkElements +
                          group * 128 + (threadIdx.x & 31) * 4;
    asm volatile("ld.global.u64 %0, [%1];" : "=l"(p.bf16) : "l"(address) : "memory");
  }
  return p;
}

__device__ __forceinline__ void fold(const Packet& p, __nv_bfloat16 (&accum)[4]) {
  if (p.quantized) taco_ma2::fold(p.fp8, accum);
  else taco_ma2::fold_local(p.bf16, accum);
}
}  // namespace taco_ma2c

__global__ void taco_ma2c_decode_kernel(const __grid_constant__ FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {
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
    *reinterpret_cast<uint2*>(output + offset) = packed;
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
  taco_ma2::store(output, group, accum);
}
