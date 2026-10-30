// Basic producer-fused TACO: materialize decoded remote contributions, then fold.
// The codec and source-order BF16 arithmetic remain the frozen fused-v2 ones.
// TACO attribution: tools/a800/phase4/TACO_LICENSE.txt.
#pragma once

// One CTA decodes one H128 group from one remote source. The source dimension
// is in original ring order, so scratch slot i corresponds to rank+i+1.
__global__ void taco_basic_decode_kernel(FluxTacoConfig c, __nv_bfloat16* decoded) {
  __shared__ flux_taco::Scratch scratch;
  const int64_t group = blockIdx.x;
  const int source_slot = blockIdx.y;
  const int src = (c.rank + source_slot + 1) % c.world;
  const int nt = (c.n + 127) / 128;
  const int row = group / nt;
  const int col = (group % nt) * 128 + threadIdx.x;
  const int64_t groups = taco_groups(c);
  const int64_t chunk = int64_t(c.m / c.world) * c.n;
  const unsigned char* packet = c.peers[c.rank] + src * taco_source_bytes(c);
  const float* quant = reinterpret_cast<const float*>(packet + groups * 128);
  const float* adaptive = quant + groups;
  // Every thread participates, including padded N tails: decode uses CTA sync.
  const float value = flux_taco::decode(packet + group * 128,
                                        quant[group], adaptive[group], scratch);
  if (col < c.n)
    decoded[int64_t(source_slot) * chunk + int64_t(row) * c.n + col] =
        __float2bfloat16(value);
}

// Deliberately ordinary scalar BF16 fold: no warp rewrite, packed loads,
// source prefetch, source-parallel addition or changed accumulation order.
__global__ void taco_basic_reduce_kernel(FluxTacoConfig c,
    const __nv_bfloat16* local, const __nv_bfloat16* decoded,
    __nv_bfloat16* output) {
  const int64_t group = blockIdx.x;
  const int nt = (c.n + 127) / 128;
  const int row = group / nt;
  const int col = (group % nt) * 128 + threadIdx.x;
  if (col >= c.n) return;
  const int64_t chunk = int64_t(c.m / c.world) * c.n;
  const int64_t index = int64_t(row) * c.n + col;
  __nv_bfloat16 accum = __float2bfloat16(0.f);
  for (int source_slot = 0; source_slot < c.world - 1; ++source_slot) {
    const auto value = decoded[int64_t(source_slot) * chunk + index];
    accum = __float2bfloat16(__bfloat162float(accum) + __bfloat162float(value));
  }
  const auto value = local[int64_t(c.rank) * chunk + index];
  accum = __float2bfloat16(__bfloat162float(accum) + __bfloat162float(value));
  output[index] = accum;
}
