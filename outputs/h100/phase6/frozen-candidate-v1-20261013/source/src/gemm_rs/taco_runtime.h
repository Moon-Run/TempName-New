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
  // Optional immutable [source][physical M-tile][N-tile] mask. Null = all remote.
  const unsigned char* selected = nullptr;
  int allow_capture = 0;
  int deferred_reuse_barrier = 0;
  // Immutable local physical tiles with at least one quantized source.
  // -1 means no compact index was supplied; zero is an all-BF16 destination.
  const int* quant_tiles = nullptr;
  int quant_tile_count = -1;
};

__host__ __device__ inline bool taco_selected(const FluxTacoConfig& c, int src,
                                             int dst, int tile) {
  const int tiles = (c.m / 128) * ((c.n + 127) / 128);
  return src != dst && (!c.selected || c.selected[src * tiles + tile] != 0);
}

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
extern "C" int taco_configure_selective(void** peers, int rank, int world, int m, int n,
                                        const unsigned char* selected);
extern "C" int taco_configure_selective_compact(void** peers, int rank, int world, int m, int n,
                                                const unsigned char* selected,
                                                const int* quant_tiles, int count);
extern "C" void taco_reset();
// Instance-owned configuration: captured during GemmRS construction and scoped
// around forward in C++, avoiding Python configure/reset on every invocation.
extern "C" int taco_instance_config_supported();
extern "C" void taco_set_config(FluxTacoConfig config);
extern "C" void taco_allow_capture();
extern "C" void taco_defer_reuse_barrier();
// Diagnostic controls affect only the measured TP4 selective M8192/N2048/K2048
// MLP. Zero keeps the registry/default scheduling; attention never uses these.
extern "C" int taco_set_gemm_tuning(int stages, int avail_sms);
extern "C" int taco_gemm_stages();
extern "C" int taco_gemm_avail_sms();
// 1: epilogue-fused encoding; 2: BF16 staging followed by standalone encoding.
extern "C" int taco_placement();
extern "C" int taco_encode_scatter(FluxTacoConfig config, const void* staged_bf16,
                                   cudaStream_t stream);
extern "C" int taco_decode_reduce(FluxTacoConfig config, const void* local_bf16,
                                   void* output_bf16, cudaStream_t stream);
