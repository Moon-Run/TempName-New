// Included after FluxTacoConfig and taco_source_bytes. Exact TP8 M2048/N4096.
#pragma once

constexpr int kOp3Fragments = 512;
constexpr int kOp3ReadyWords = 8 * kOp3Fragments;
constexpr int kOp3DoneWords = 8192;
constexpr int kOp3EarlyCtas = 64;
constexpr int kOp3MaxTryGroups = kOp3EarlyCtas * 16;
// A 32-bit stop epoch plus padding keeps all diagnostic atomics 8-byte aligned.
constexpr int kOp3StatsOffset = (kOp3ReadyWords + kOp3DoneWords + 2) * sizeof(uint32_t);
constexpr int kOp3StatsCount = 7;
constexpr int kOp3ControlBytes = kOp3StatsOffset + kOp3StatsCount * sizeof(uint64_t);
static_assert(kOp3ControlBytes == 49216);

__host__ __device__ inline uint32_t* taco_op3_ready(const FluxTacoConfig& c, int dst) {
  return reinterpret_cast<uint32_t*>(c.peers[dst] + dst * taco_source_bytes(c));
}
__host__ __device__ inline uint32_t* taco_op3_done(const FluxTacoConfig& c) {
  return taco_op3_ready(c, c.rank) + kOp3ReadyWords;
}
__host__ __device__ inline uint32_t* taco_op3_stop_ptr(const FluxTacoConfig& c) {
  return taco_op3_done(c) + kOp3DoneWords;
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
__device__ __forceinline__ bool taco_op3_stopped(const FluxTacoConfig& c) {
  uint32_t value;
  auto* p = taco_op3_stop_ptr(c);
  asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(value) : "l"(p) : "memory");
  return value == c.op3_epoch;
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
__device__ __forceinline__ void taco_op3_publish(const FluxTacoConfig& c,
                                                int dst, int tile, int step) {
  // Every writer participates: stores -> CTA barrier -> leader system release.
  // PTX 8.7 sections 8.9.4/8.9.5 define the transitive synchronization chain.
  // CUDA 12.8 "Memory Fence Interference" explains cumulative system releases.
  // The barrier is essential; this is NOT a fence by an unsynchronized leader.
  __syncthreads();
  if (threadIdx.x == 0) {
    const int fragment = (tile - dst * 64) * 8 + step;
    taco_op3_release(taco_op3_ready(c, dst) + c.rank * kOp3Fragments + fragment, c.op3_epoch);
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
__device__ __forceinline__ int taco_op3_group(int fragment, int compact) {
  const int tile = fragment / 8, step = fragment % 8;
  const int row = (tile / 32) * 128 + step * 8 + compact % 8 + (compact / 8) * 64;
  return row * 32 + tile % 32;
}
#endif

extern "C" int taco_op3_protocol_version();
extern "C" int taco_op3_configure(uint32_t epoch, int overlap, int rounds, int diagnostics);
extern "C" int taco_op3_prepare(FluxTacoConfig config, cudaStream_t stream);
extern "C" int taco_op3_try(FluxTacoConfig config, const void* local, void* output,
                           cudaStream_t stream);
extern "C" int taco_op3_tail(FluxTacoConfig config, const void* local, void* output,
                            cudaStream_t stream);
extern "C" int taco_op3_stop(FluxTacoConfig config, cudaStream_t stream);
