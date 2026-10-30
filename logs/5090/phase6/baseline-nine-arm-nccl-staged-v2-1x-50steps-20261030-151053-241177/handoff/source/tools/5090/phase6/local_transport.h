// Experimental RTX5090 local TACO producer + compact NCCL staging ABI.
// All tensor storage is owned by the Python caller and belongs to one GPU.
#pragma once
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

int rtx5090_local_protocol_version();
const char* rtx5090_local_last_error();
int rtx5090_local_create(void** handle, int rank, int k_local, int cache_submission,
    const void* selected, void* tx_bf16, void* tx_packets,
    void* rx_bf16, void* rx_packets, const void* tx_offsets, const void* rx_offsets,
    void* tx_wire, void* rx_wire, int64_t tx_bytes, int64_t rx_bytes, void* stream);
int rtx5090_local_reconfigure(void* handle, void* stream);
int64_t rtx5090_local_reconfiguration_count(void* handle);
// Private cudaMalloc allocation; excluded from PyTorch allocator statistics.
int64_t rtx5090_local_workspace_bytes(void* handle);
int rtx5090_local_produce(void* handle, const void* input, const void* weight, void* stream);
// Python must order NCCL receive completion onto this same creation stream
// before calling decode. The native decoder then consumes only local buffers.
int rtx5090_local_decode(void* handle, void* output,
    const void* bias, const void* residual, void* stream);
// [version, initializations, producer_calls, decoder_calls, consumer_calls,
//  phase (0=ready,1=produced), failed, k_local].
int rtx5090_local_state(void* handle, int64_t* values, int length);
// Final object cleanup only; waits for the creation stream before releasing
// its private CUTLASS workspace. It never frees caller tensor buffers.
int rtx5090_local_destroy(void* handle);

#ifdef __cplusplus
}
#endif
