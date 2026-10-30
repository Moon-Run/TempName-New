// TACO/COCCL adaptation. See tools/a800/phase4/TACO_LICENSE.txt (BSD-3-Clause).
#pragma once
#include <cstdint>
#include <cuda_runtime_api.h>

// Experimental, exact-shape SM80 protocol. No change to public Flux argument ABI.
struct FluxTacoConfig {
  unsigned char* peers[8]{};
  int enabled = 0;
  int rank = 0;
  int world = 0;
  int m = 0;
  int n = 0;
};

// One source slot: padded FP8 rows, then FP32 quant scales, then adaptive scales.
// All Hadamard coefficients, including N-tail padding, must cross the wire.
__host__ __device__ inline int64_t taco_groups(const FluxTacoConfig& c) {
  return int64_t(c.m / c.world) * ((c.n + 127) / 128);
}
__host__ __device__ inline int64_t taco_source_bytes(const FluxTacoConfig& c) {
  return taco_groups(c) * 136;
}

extern "C" FluxTacoConfig taco_config();
extern "C" int taco_configure(void** peers, int rank, int world, int m, int n);
extern "C" void taco_reset();
// 1: epilogue-fused encoding; 2: BF16 staging followed by standalone encoding.
extern "C" int taco_placement();
extern "C" int taco_encode_scatter(FluxTacoConfig config, const void* staged_bf16,
                                   cudaStream_t stream);
extern "C" int taco_decode_reduce(FluxTacoConfig config, const void* local_bf16,
                                   void* output_bf16, cudaStream_t stream);
