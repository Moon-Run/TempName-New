// MA12: destination-local fragment publication and slot-completion epochs.
// The packet format and arithmetic are inherited from the original MA89 path.
#pragma once
#include "gemm_rs/taco_runtime.h"

namespace taco_ma12 {
constexpr int kTilesPerRank = 64;
constexpr int kFragments = 8;
constexpr int kReadyWords = 8 * kTilesPerRank * kFragments;
constexpr int kWords = kReadyWords + 1;

__host__ __device__ constexpr int ready_index(int source, int local_tile, int fragment) {
  return (source * kTilesPerRank + local_tile) * kFragments + fragment;
}

__host__ __device__ constexpr int consumer_ready_index(int source, int group) {
  const int row = group / 32;
  return ready_index(source, (row / 128) * 32 + group % 32, (row % 64) / 8);
}

__device__ __forceinline__ int acquire(const int* address) {
  int value;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];"
               : "=r"(value) : "l"(address) : "memory");
  return value;
}

__device__ __forceinline__ void release(int* address, int value) {
  asm volatile("st.release.sys.global.u32 [%0], %1;"
               :: "l"(address), "r"(value) : "memory");
}

// All 128 producer threads participate. Each thread first publishes its own
// packet/BF16 stores to system scope; only then can thread zero publish ready.
// A Stream-K callback can own one fragment, so publication is in end_step,
// never end_epilogue and never dependent on the other fragments being resident.
__device__ __forceinline__ void publish_fragment(
    const FluxTacoConfig& c, int destination, int physical_tile, int step) {
  if (!c.ma12_ready[0]) return;
  __threadfence_system();
  __syncthreads();
  if (threadIdx.x == 0) {
    const int local_tile = physical_tile - destination * kTilesPerRank;
    const int index = ready_index(c.rank, local_tile, step);
    release(c.ma12_ready[destination] + index, c.ma12_epoch);
  }
}

// One warp consumes one H128 row. Its source order remains exactly MA89's.
// The acquire in lane zero followed by the warp barrier orders subsequent
// packet loads in all lanes. This is called before each source's first load.
__device__ __forceinline__ void wait_source(
    const FluxTacoConfig& c, int group, int source) {
  const int index = consumer_ready_index(source, group);
  if ((threadIdx.x & 31) == 0) {
    while (acquire(c.ma12_ready[c.rank] + index) < c.ma12_epoch) __nanosleep(64);
  }
  __syncwarp();
}
}  // namespace taco_ma12
