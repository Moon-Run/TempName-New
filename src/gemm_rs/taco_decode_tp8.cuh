// TACO adaptation; see tools/a800/phase4/TACO_LICENSE.txt.
// Isolated TP8 decoder experiment. Included by tp8_quant/rebuild_decoder.py.
// Exact shape: M=N=2048, TP=8. Preserve source ring order, H128 codec,
// BF16 rounding, and all publication/workspace lifetime barriers.
#pragma once

// One launch contains disjoint dense and mixed work. Dense CTAs skip the exact
// union of selected tiles. In a mixed CTA, eight warps decode/load the eight
// sources concurrently, then warp zero applies the unchanged BF16 ring sum.
__global__ void taco_decode_cooperative8_kernel(FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {
  constexpr int chunk = 524288, groups = 4096;
  if (blockIdx.x < 256) {
    const int offset = (blockIdx.x * 256 + threadIdx.x) * 8;
    const int tile = (c.rank * 2 + offset / (2048 * 128)) * 16 + (offset % 2048) / 128;
    bool mixed = false;
    #pragma unroll
    for (int src = 0; src < 8; ++src)
      mixed |= src != c.rank && c.selected[src * 256 + tile];
    if (mixed) return;
    uint4 packed = make_uint4(0, 0, 0, 0);
    auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
    #pragma unroll
    for (int i = 1; i <= 8; ++i) {
      const int src = (c.rank + i) & 7;
      uint4 value = *reinterpret_cast<const uint4*>(local + src * chunk + offset);
      auto* x = reinterpret_cast<__nv_bfloat162*>(&value);
      #pragma unroll
      for (int j = 0; j < 4; ++j) sum[j] = __hadd2(sum[j], x[j]);
    }
    *reinterpret_cast<uint4*>(output + offset) = packed;
    return;
  }
  __shared__ __nv_bfloat16 contributions[8][128];
  const int work = blockIdx.x - 256;
  const int tile = c.quant_tiles[work / 128];
  const int row = (tile / 16) * 128 + work % 128, col = tile % 16;
  const int src = threadIdx.x / 32, lane = threadIdx.x & 31;
  const int group = row * 16 + col;
  float values[4];
  if (src != c.rank && c.selected[src * 256 + c.rank * 32 + tile]) {
    const unsigned char* slot = c.peers[c.rank] + src * int64_t(groups) * 136;
    const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
    const float* as = qs + groups;
    flux_taco::decode_warp(slot + group * 128, qs[group], as[group], values);
  } else {
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      values[j] = __bfloat162float(local[src * chunk + row * 2048 + col * 128 + lane + j * 32]);
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j) contributions[src][lane+j*32] = __float2bfloat16(values[j]);
  __syncthreads();
  if (src == 0) {
    __nv_bfloat16 accum[4] = {};
    #pragma unroll
    for (int i = 1; i <= 8; ++i) {
      const int source = (c.rank + i) & 7;
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        accum[j] = __hadd(accum[j], contributions[source][lane+j*32]);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) output[row*2048+col*128+lane+j*32] = accum[j];
  }
}

// Materialize only selected source/tile pairs into this destination's BF16
// workspace after the publication barrier. Each source has a disjoint slot;
// the epilogue wrote FP8 instead of BF16 there. The subsequent reduction and
// the next invocation run on the same stream, and workspace reuse still uses
// the existing double-buffered protocol.
__global__ void taco_materialize8_kernel(FluxTacoConfig c, __nv_bfloat16* local) {
  constexpr int groups = 4096, chunk = 524288;
  const int tile = c.quant_tiles[blockIdx.x / 32], src = blockIdx.y;
  if (src == c.rank || !c.selected[src * 256 + c.rank * 32 + tile]) return;
  const int row = (tile / 16) * 128 + (blockIdx.x % 32) * 4 + threadIdx.x / 32;
  const int col = tile % 16, group = row * 16 + col, lane = threadIdx.x & 31;
  const unsigned char* slot = c.peers[c.rank] + src * int64_t(groups) * 136;
  const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
  const float* as = qs + groups;
  float values[4];
  flux_taco::decode_warp(slot + group * 128, qs[group], as[group], values);
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    local[src * chunk + row * 2048 + col * 128 + lane + j * 32] = __float2bfloat16(values[j]);
}

__global__ void taco_reduce_materialized8_kernel(FluxTacoConfig c,
    const __nv_bfloat16* local, __nv_bfloat16* output) {
  constexpr int chunk = 524288;
  const int offset = (blockIdx.x * 256 + threadIdx.x) * 8;
  uint4 packed = make_uint4(0, 0, 0, 0);
  auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
  #pragma unroll
  for (int i = 1; i <= 8; ++i) {
    const int src = (c.rank + i) & 7;
    uint4 value = *reinterpret_cast<const uint4*>(local + src * chunk + offset);
    auto* x = reinterpret_cast<__nv_bfloat162*>(&value);
    #pragma unroll
    for (int j = 0; j < 4; ++j) sum[j] = __hadd2(sum[j], x[j]);
  }
  *reinterpret_cast<uint4*>(output + offset) = packed;
}

__global__ void taco_decode_ring8_kernel(FluxTacoConfig c, const __nv_bfloat16* local,
                                        __nv_bfloat16* output) {
  // Each warp owns an H128 group. Four groups per CTA avoid cross-warp
  // scratch/barriers, including on the common unquantized selective path.
  const int lane = threadIdx.x & 31;
  const int world = 8;
  const int n = 2048;
  const int m = 2048;
  const int group = blockIdx.x * 4 + threadIdx.x / 32;
  const int nt = (n + 127) / 128;
  const int64_t groups = int64_t(m / world) * nt, source_bytes = groups * 136;
  if (group >= groups) return;  // Uniform within a warp.
  const int row = group / nt, col = (group % nt) * 128 + lane;
  const int64_t chunk = int64_t(m / world) * n;
  const int physical_tile = ((c.rank * (m / world) + row) / 128) * nt + group % nt;
  bool any_quant = !c.selected;
  if (c.selected) {
    #pragma unroll
    for (int src = 0; src < (8); ++src)
      any_quant |= src != c.rank && c.selected[src * (m / 128) * nt + physical_tile];
  }
  if (!any_quant && (group % nt + 1) * 128 <= n) {
    // Most sparse-mask groups are entirely BF16. Four adjacent elements per
    // lane give one aligned 64-bit transaction and two packed BF16 additions.
    const int64_t offset = int64_t(row) * n + (group % nt) * 128 + lane * 4;
    uint2 packed = make_uint2(0, 0);
    auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
    #pragma unroll
    for (int i = 1; i <= (8); ++i) {
      int src = c.rank + i;
      if (src >= world) src -= world;
      uint2 value = *reinterpret_cast<const uint2*>(local + src * chunk + offset);
      auto* x = reinterpret_cast<__nv_bfloat162*>(&value);
      sum[0] = __hadd2(sum[0], x[0]);
      sum[1] = __hadd2(sum[1], x[1]);
    }
    *reinterpret_cast<uint2*>(output + offset) = packed;
    return;
  }
  __nv_bfloat16 accum[4] = {};
  // Match Flux ring_reduction=True: rank+1,...,rank with BF16 rounding
  // after decoding each source and after every addition.
  #pragma unroll
  for (int i = 1; i <= (8); ++i) {
    int src = c.rank + i;
    if (src >= world) src -= world;
    float value[4];
    if (!taco_selected(c, src, c.rank, physical_tile)) {
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        value[j] = col + j * 32 < n ?
            __bfloat162float(local[src * chunk + int64_t(row) * n + col + j * 32]) : 0.f;
    } else {
      const unsigned char* slot = c.peers[c.rank] + src * source_bytes;
      const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
      const float* as = qs + groups;
      flux_taco::decode_warp(slot + group * 128, qs[group], as[group], value);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      accum[j] = __float2bfloat16(__bfloat162float(accum[j]) +
                                __bfloat162float(__float2bfloat16(value[j])));
  }
  #pragma unroll
  for (int j = 0; j < 4; ++j)
    if (col + j * 32 < n) output[int64_t(row) * n + col + j * 32] = accum[j];
}

template <int Rows, bool RowMajor = false>
__global__ void taco_decode_tile8_kernel(FluxTacoConfig c, const __nv_bfloat16* local,
                                        __nv_bfloat16* output) {
  constexpr int world = 8;
  const int lane = threadIdx.x & 31, warp = threadIdx.x / 32;
  const int nt = 2048 / 128;
  const int tile = blockIdx.x / (128 / Rows);
  const int col_tile = RowMajor ? blockIdx.x : tile % nt;
  const int first_row = RowMajor ? blockIdx.y * Rows :
      (tile / nt) * 128 + (blockIdx.x % (128 / Rows)) * Rows;
  const int physical_tile = (c.rank * (2048 / (128 * world)) + first_row / 128) * nt + col_tile;
  const int tiles = (2048 / 128) * nt;
  const int64_t chunk = int64_t(2048 / world) * 2048;
  const int64_t groups = int64_t(2048 / world) * nt;
  bool selected[world];
  bool any_quant = false;
  #pragma unroll
  for (int src = 0; src < world; ++src) {
    selected[src] = src != c.rank && c.selected[src * tiles + physical_tile];
    any_quant |= selected[src];
  }
  if (!any_quant) {
    #pragma unroll 1
    for (int row = first_row + warp; row < first_row + Rows; row += 4) {
      const int64_t offset = int64_t(row) * 2048 + col_tile * 128 + lane * 4;
      uint2 packed = make_uint2(0, 0);
      auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
      #pragma unroll
      for (int i = 1; i <= world; ++i) {
        const int src = (c.rank + i) & 7;
        uint2 value = *reinterpret_cast<const uint2*>(local + src * chunk + offset);
        auto* x = reinterpret_cast<__nv_bfloat162*>(&value);
        sum[0] = __hadd2(sum[0], x[0]);
        sum[1] = __hadd2(sum[1], x[1]);
      }
      *reinterpret_cast<uint2*>(output + offset) = packed;
    }
    return;
  }
  #pragma unroll 1
  for (int row = first_row + warp; row < first_row + Rows; row += 4) {
    const int64_t group = int64_t(row) * nt + col_tile;
    __nv_bfloat16 accum[4] = {};
    #pragma unroll
    for (int i = 1; i <= world; ++i) {
      const int src = (c.rank + i) & 7;
      float value[4];
      if (!selected[src]) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          value[j] = __bfloat162float(local[src * chunk + int64_t(row) * 2048 +
                                           col_tile * 128 + lane + j * 32]);
      } else {
        const unsigned char* slot = c.peers[c.rank] + src * groups * 136;
        const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
        const float* as = qs + groups;
        flux_taco::decode_warp(slot + group * 128, qs[group], as[group], value);
      }
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        accum[j] = __float2bfloat16(__bfloat162float(accum[j]) +
                                  __bfloat162float(__float2bfloat16(value[j])));
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      output[int64_t(row) * 2048 + col_tile * 128 + lane + j * 32] = accum[j];
  }
}

template <int Threads>
__global__ void taco_decode_flat8_kernel(FluxTacoConfig c, const __nv_bfloat16* local,
                                        __nv_bfloat16* output) {
  constexpr int n = 2048, tiles = 256, groups = 4096, chunk = 524288;
  const int lane = threadIdx.x & 31;
  const int offset = (blockIdx.x * Threads + threadIdx.x) * 8;
  const int row = offset / n, col_pair = ((offset % n) / 128) & ~1;
  const int physical_tile = (c.rank * 2 + row / 128) * 16 + col_pair;
  bool selected[2][8];
  bool any_quant = false;
  #pragma unroll
  for (int half = 0; half < 2; ++half) {
    #pragma unroll
    for (int src = 0; src < 8; ++src) {
      selected[half][src] = src != c.rank && c.selected[src * tiles + physical_tile + half];
      any_quant |= selected[half][src];
    }
  }
  if (!any_quant) {
    uint4 packed = make_uint4(0, 0, 0, 0);
    auto* sum = reinterpret_cast<__nv_bfloat162*>(&packed);
    #pragma unroll
    for (int i = 1; i <= 8; ++i) {
      const int src = (c.rank + i) & 7;
      uint4 value = *reinterpret_cast<const uint4*>(local + src * chunk + offset);
      auto* x = reinterpret_cast<__nv_bfloat162*>(&value);
      #pragma unroll
      for (int j = 0; j < 4; ++j) sum[j] = __hadd2(sum[j], x[j]);
    }
    *reinterpret_cast<uint4*>(output + offset) = packed;
    return;
  }
  #pragma unroll
  for (int half = 0; half < 2; ++half) {
    const int col_tile = col_pair + half, group = row * 16 + col_tile;
    __nv_bfloat16 accum[4] = {};
    #pragma unroll
    for (int i = 1; i <= 8; ++i) {
      const int src = (c.rank + i) & 7;
      float value[4];
      if (!selected[half][src]) {
        #pragma unroll
        for (int j = 0; j < 4; ++j)
          value[j] = __bfloat162float(local[src * chunk + row * n + col_tile * 128 + lane + j * 32]);
      } else {
        const unsigned char* slot = c.peers[c.rank] + src * int64_t(groups) * 136;
        const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
        const float* as = qs + groups;
        flux_taco::decode_warp(slot + group * 128, qs[group], as[group], value);
      }
      #pragma unroll
      for (int j = 0; j < 4; ++j)
        accum[j] = __float2bfloat16(__bfloat162float(accum[j]) +
                                  __bfloat162float(__float2bfloat16(value[j])));
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j)
      output[row * n + col_tile * 128 + lane + j * 32] = accum[j];
  }
}
