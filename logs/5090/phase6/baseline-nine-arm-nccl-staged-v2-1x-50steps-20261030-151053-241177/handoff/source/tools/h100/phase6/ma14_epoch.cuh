// MA14 coarse publication epochs. Producer, codec and decoder are unchanged.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>
#include "gemm_rs/taco_runtime.h"

// Receiver-owned slots have one writer per source. A release/acquire pair
// orders all preceding same-stream GEMM writes, including remote BF16 stores,
// FP8 payload and both scale arrays. The values are never reset during reuse.
__device__ __forceinline__ void taco_ma14_release(unsigned long long* address,
                                                 unsigned long long value) {
  asm volatile("st.release.sys.global.u64 [%0], %1;" ::
               "l"(address), "l"(value) : "memory");
}

__device__ __forceinline__ unsigned long long
taco_ma14_acquire(const unsigned long long* address) {
  unsigned long long value;
  asm volatile("ld.acquire.sys.global.u64 %0, [%1];" : "=l"(value) :
               "l"(address) : "memory");
  return value;
}

// Exactly one small CTA replaces the old CudaIpcBarrierAllKernel. This is a
// full publication rendezvous, not fragment publication or a persistent
// consumer: every producer is already ahead of its rendezvous on its stream.
__global__ void taco_ma14_epoch_barrier_kernel(const __grid_constant__ Ma14EpochConfig c) {
  const int peer = threadIdx.x;
  if (peer < 8) taco_ma14_release(c.peers[peer] + c.rank, c.epoch);
  __syncthreads();
  if (peer < 8) {
    while (taco_ma14_acquire(c.peers[c.rank] + peer) < c.epoch)
      __nanosleep(64);
  }
}

extern "C" int taco_ma14_epoch_protocol() { return 1402; }

extern "C" int taco_ma14_epoch_barrier(Ma14EpochConfig c, cudaStream_t stream) {
  if (c.rank < 0 || c.rank >= 8 || c.epoch == 0 ||
      c.epoch > 0x7fffffffffffffffULL) return int(cudaErrorInvalidValue);
  for (int rank = 0; rank < 8; ++rank) {
    if (!c.peers[rank] || reinterpret_cast<uintptr_t>(c.peers[rank]) % 8)
      return int(cudaErrorInvalidValue);
  }
  taco_ma14_epoch_barrier_kernel<<<1, 32, 0, stream>>>(c);
  return int(cudaGetLastError());
}
