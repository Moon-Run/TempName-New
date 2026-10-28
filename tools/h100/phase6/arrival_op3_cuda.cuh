// Op3 readiness protocol for the pinned all-remote H128/E4M3 implementation.
// Included after FluxTacoConfig and taco_source_bytes. No selective-mask ABI.
#pragma once

constexpr int kOp3Fragments = 64 * 8;
constexpr int kOp3ReadyWords = 8 * kOp3Fragments;
constexpr int kOp3StatsOffset = (kOp3ReadyWords + kOp3Fragments) * sizeof(uint32_t);
constexpr int kOp3StatsCount = 5;
constexpr int kOp3ControlBytes = kOp3StatsOffset + kOp3StatsCount * sizeof(uint64_t);

// The packet slot for source == destination is otherwise unused: its values
// always come from the local BF16 buffer. Other source slots are unchanged.
__host__ __device__ inline uint32_t* taco_op3_ready(const FluxTacoConfig& c, int dst) {
  return reinterpret_cast<uint32_t*>(c.peers[dst] + dst * taco_source_bytes(c));
}
__host__ __device__ inline uint32_t* taco_op3_done(const FluxTacoConfig& c) {
  return taco_op3_ready(c, c.rank) + kOp3ReadyWords;
}
__host__ __device__ inline unsigned long long* taco_op3_stats(const FluxTacoConfig& c) {
  return reinterpret_cast<unsigned long long*>(
      reinterpret_cast<unsigned char*>(taco_op3_ready(c, c.rank)) + kOp3StatsOffset);
}

#ifdef __CUDACC__
__device__ __forceinline__ uint32_t taco_op3_acquire(const uint32_t* p) {
  uint32_t value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return value;
}
__device__ __forceinline__ void taco_op3_release(uint32_t* p, uint32_t value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;" :: "l"(p), "r"(value) : "memory");
}
__device__ __forceinline__ unsigned long long taco_op3_clock() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__device__ __forceinline__ void taco_op3_first(unsigned long long* p,
                                              unsigned long long time) {
  auto previous = atomicCAS(p, 0ULL, time);
  if (previous != 0) atomicMin(p, time);
}
// ALL store threads call this. A fence in thread zero alone is insufficient.
// This is a fragment publication, never a wait for another CTA's fragment.
__device__ __forceinline__ void taco_op3_publish(const FluxTacoConfig& c,
                                                int dst, int tile, int step) {
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int local_tile = tile - dst * 64;
    taco_op3_release(taco_op3_ready(c, dst) + c.rank * kOp3Fragments +
                         local_tile * 8 + step, c.op3_epoch);
    if (c.op3_diagnostics)
      atomicMax(taco_op3_stats(c) + 1, taco_op3_clock());
  }
}
__device__ __forceinline__ bool taco_op3_is_ready(const FluxTacoConfig& c, int fragment) {
  const auto* ready = taco_op3_ready(c, c.rank);
  for (int source = 0; source < 8; ++source)
    if (taco_op3_acquire(ready + source * kOp3Fragments + fragment) != c.op3_epoch)
      return false;
  return true;
}
#endif

extern "C" int taco_op3_configure(uint32_t epoch, int overlap, int rounds, int diagnostics);
extern "C" int taco_op3_prepare(FluxTacoConfig config, cudaStream_t stream);
extern "C" int taco_op3_try(FluxTacoConfig config, const void* local, void* output,
                           cudaStream_t stream);
extern "C" int taco_op3_tail(FluxTacoConfig config, const void* local, void* output,
                            cudaStream_t stream);
