// TACO adaptation; see tools/a800/phase4/TACO_LICENSE.txt.
#include "gemm_rs/taco_codec.cuh"

static thread_local FluxTacoConfig active;
extern "C" FluxTacoConfig taco_config() { return active; }
extern "C" void taco_reset() { active = FluxTacoConfig{}; }
extern "C" int taco_placement() {
#ifdef FLUX_TACO_SEPARATE
  return 2;
#else
  return 1;
#endif
}
extern "C" int taco_configure(void** peers, int rank, int world, int m, int n) {
  if (active.enabled || !peers || (world != 2 && world != 4 && world != 8) ||
      rank < 0 || rank >= world || m <= 0 || m % (128 * world) || n <= 0 || n % 8)
    return int(cudaErrorInvalidValue);
  FluxTacoConfig c;
  c.rank = rank; c.world = world; c.m = m; c.n = n;
  for (int r = 0; r < world; ++r) {
    if (!peers[r]) return int(cudaErrorInvalidValue);
    c.peers[r] = static_cast<unsigned char*>(peers[r]);
  }
  c.enabled = 1; active = c;
  return int(cudaSuccess);
}

__global__ void taco_encode_scatter_kernel(FluxTacoConfig c, const __nv_bfloat16* staged) {
  __shared__ flux_taco::Scratch scratch;
  const int64_t groups = taco_groups(c);
  const int dst = blockIdx.x / groups;
  if (dst == c.rank) return;  // Local contribution stays BF16 in its original slot.
  const int64_t group = blockIdx.x % groups;
  const int nt = (c.n + 127) / 128;
  const int row = group / nt;
  const int col = (group % nt) * 128 + threadIdx.x;
  const int global_row = dst * (c.m / c.world) + row;
  float value = col < c.n ? __bfloat162float(staged[int64_t(global_row)*c.n+col]) : 0.f;
  unsigned char* slot = c.peers[dst] + c.rank*taco_source_bytes(c);
  float* qs = reinterpret_cast<float*>(slot + groups*128);
  float* as = qs + groups;
  const int valid = min(128, c.n - int(group % nt)*128);
  flux_taco::encode(value, valid, slot+group*128, qs+group, as+group, scratch);
}

extern "C" int taco_encode_scatter(FluxTacoConfig c, const void* staged, cudaStream_t stream) {
  if (!c.enabled || !staged) return int(cudaErrorInvalidValue);
  taco_encode_scatter_kernel<<<taco_groups(c)*c.world, 128, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(staged));
  return int(cudaGetLastError());
}

__global__ void taco_decode_ring_kernel(FluxTacoConfig c, const __nv_bfloat16* local,
                                        __nv_bfloat16* output) {
  __shared__ flux_taco::Scratch scratch;
  const int64_t group = blockIdx.x;
  const int nt = (c.n + 127) / 128;
  const int row = group / nt, col = (group % nt) * 128 + threadIdx.x;
  const int64_t groups = taco_groups(c), source_bytes = taco_source_bytes(c);
  const int64_t chunk = int64_t(c.m / c.world) * c.n;
  __nv_bfloat16 accum = __float2bfloat16(0.f);
  // Match Flux ring_reduction=True: rank+1,...,rank with BF16 rounding
  // after decoding each source and after every addition.
  for (int i = 1; i <= c.world; ++i) {
    int src = (c.rank + i) % c.world;
    float value;
    if (src == c.rank) {
      value = col < c.n ? __bfloat162float(local[src * chunk + int64_t(row) * c.n + col]) : 0.f;
    } else {
      const unsigned char* slot = c.peers[c.rank] + src * source_bytes;
      const float* qs = reinterpret_cast<const float*>(slot + groups * 128);
      const float* as = qs + groups;
      value = flux_taco::decode(slot + group * 128, qs[group], as[group], scratch);
    }
    accum = __float2bfloat16(__bfloat162float(accum) + __bfloat162float(__float2bfloat16(value)));
    __syncthreads();
  }
  if (col < c.n) output[int64_t(row) * c.n + col] = accum;
}

extern "C" int taco_decode_reduce(FluxTacoConfig c, const void* local, void* output,
                                   cudaStream_t stream) {
  if (!c.enabled || !local || !output) return int(cudaErrorInvalidValue);
  taco_decode_ring_kernel<<<taco_groups(c), 128, 0, stream>>>(
      c, static_cast<const __nv_bfloat16*>(local), static_cast<__nv_bfloat16*>(output));
  return int(cudaGetLastError());
}
