// TACO arithmetic and decoder remain in their frozen source translation units.
// This file owns local execution, lossless byte packing and lifecycle checks.
#include "gemm_rs/local_transport.h"
#include "gemm_rs/local_layout.h"
#include "gemm_rs/taco_runtime.h"
#include "flux/args/gemm_rs.h"
#include "flux/op_registry.h"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_set>

#ifndef RTX5090_LOCAL_CONSUMER
#define RTX5090_LOCAL_CONSUMER 0
#endif
#ifndef RTX5090_LOCAL_SELECTIVE
#define RTX5090_LOCAL_SELECTIVE 0
#endif

// Each frozen arm has its own FluxTacoConfig ABI; the private setter is added
// beside that arm's original TLS object, without modifying decoder arithmetic.
extern "C" void rtx5090_local_set_taco_config(FluxTacoConfig config);

namespace {
using namespace bytedance::flux;
using namespace rtx5090_local_layout;
thread_local std::string last_error;

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
void cuda_check(cudaError_t code) {
  if (code != cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void local_pointer(const void* pointer, int device, bool nullable = false) {
  if (!pointer && nullable) return;
  check(pointer != nullptr, "Missing local CUDA buffer");
  cudaPointerAttributes attributes{};
  cuda_check(cudaPointerGetAttributes(&attributes, pointer));
  check(attributes.type == cudaMemoryTypeDevice && attributes.device == device,
        "Every staging/input pointer must be device memory on the creation GPU");
  check(reinterpret_cast<uintptr_t>(pointer) % 8 == 0, "CUDA buffer must be 8-byte aligned");
}

// Packet bytes are copied without any FP arithmetic. A quantized tile contains
// 128 H128 rows, then their original FP32 quant scales and adaptive scales.
__global__ void rtx5090_pack_tiles_kernel(int rank, const unsigned char* selected,
    const __nv_bfloat16* bf16, const unsigned char* packets,
    const int64_t* offsets, unsigned char* wire) {
  const int tile = blockIdx.x, destination = tile / kLocalTiles;
  if (destination == rank) return;
  const int within = tile % kLocalTiles, row0 = (within / 32) * 128, col0 = (within % 32) * 128;
  const bool quantized = !selected || selected[rank * kTiles + tile];
  unsigned char* output = wire + offsets[tile];
  if (!quantized) {
    auto* out = reinterpret_cast<uint16_t*>(output);
    const auto* in = reinterpret_cast<const uint16_t*>(bf16);
    for (int element = threadIdx.x; element < 128 * 128; element += blockDim.x)
      out[element] = in[destination * kChunk + int64_t(row0 + element / 128) * kN + col0 + element % 128];
  } else {
    const unsigned char* input = packets + destination * kPacket;
    for (int element = threadIdx.x; element < 128 * 128; element += blockDim.x) {
      const int group = (row0 + element / 128) * 32 + within % 32;
      output[element] = input[group * 128 + element % 128];
    }
    if (threadIdx.x < 128) {
      const int group = (row0 + threadIdx.x) * 32 + within % 32;
      // Copy IEEE bits, including sign and exceptional values, verbatim.
      reinterpret_cast<uint32_t*>(output + 16384)[threadIdx.x] =
          reinterpret_cast<const uint32_t*>(input + kGroups * 128)[group];
      reinterpret_cast<uint32_t*>(output + 16896)[threadIdx.x] =
          reinterpret_cast<const uint32_t*>(input + kGroups * 132)[group];
    }
  }
}

__global__ void rtx5090_unpack_tiles_kernel(int rank, const unsigned char* selected,
    __nv_bfloat16* bf16, unsigned char* packets,
    const int64_t* offsets, const unsigned char* wire) {
  const int tile = blockIdx.x, source = tile / kLocalTiles;
  if (source == rank) return;  // The producer already wrote the local BF16 contribution.
  const int within = tile % kLocalTiles, row0 = (within / 32) * 128, col0 = (within % 32) * 128;
  const bool quantized = !selected || selected[source * kTiles + rank * kLocalTiles + within];
  const unsigned char* input = wire + offsets[tile];
  if (!quantized) {
    const auto* in = reinterpret_cast<const uint16_t*>(input);
    auto* out = reinterpret_cast<uint16_t*>(bf16);
    for (int element = threadIdx.x; element < 128 * 128; element += blockDim.x)
      out[source * kChunk + int64_t(row0 + element / 128) * kN + col0 + element % 128] = in[element];
  } else {
    unsigned char* output = packets + source * kPacket;
    for (int element = threadIdx.x; element < 128 * 128; element += blockDim.x) {
      const int group = (row0 + element / 128) * 32 + within % 32;
      output[group * 128 + element % 128] = input[element];
    }
    if (threadIdx.x < 128) {
      const int group = (row0 + threadIdx.x) * 32 + within % 32;
      reinterpret_cast<uint32_t*>(output + kGroups * 128)[group] =
          reinterpret_cast<const uint32_t*>(input + 16384)[threadIdx.x];
      reinterpret_cast<uint32_t*>(output + kGroups * 132)[group] =
          reinterpret_cast<const uint32_t*>(input + 16896)[threadIdx.x];
    }
  }
}

struct Local {
  int rank, k, device, phase = 0;
  bool cache, failed = false;
  cudaStream_t stream;
  const unsigned char* selected;
  void *tx_bf16, *tx_packets, *rx_bf16, *rx_packets, *tx_wire, *rx_wire;
  const int64_t *tx_offsets, *rx_offsets;
  FluxTacoConfig producer{}, receiver{};
  std::array<void*, kWorld> outputs{}, unused{};
  OpRegistry::OpPtr cached_op;
  void* workspace = nullptr;
  size_t workspace_bytes = 0;
  int64_t initializations = 0, produces = 0, decodes = 0, consumers = 0, reconfigurations = 0;
  std::mutex mutex;

  void check_stream(void* given) const {
    int current = -1;
    cuda_check(cudaGetDevice(&current));
    check(current == device && reinterpret_cast<cudaStream_t>(given) == stream,
          "LocalTaco requires its creation GPU and stream");
    cudaStreamCaptureStatus capture;
    cuda_check(cudaStreamIsCapturing(stream, &capture));
    check(capture == cudaStreamCaptureStatusNone, "LocalTaco NCCL staging does not support CUDA Graph capture");
    check(!failed, "LocalTaco cannot retry a failed transaction");
  }

  void configure() {
    producer = FluxTacoConfig{};
    producer.enabled = 1;
    producer.rank = rank; producer.world = kWorld; producer.m = kM; producer.n = kN;
#if RTX5090_LOCAL_SELECTIVE
    producer.selected = selected;
    producer.deferred_reuse_barrier = 1;
#endif
    for (int destination = 0; destination < kWorld; ++destination) {
      producer.peers[destination] = static_cast<unsigned char*>(tx_packets) + destination * kPacket;
      outputs[destination] = static_cast<__nv_bfloat16*>(
          destination == rank ? rx_bf16 : tx_bf16) + destination * kChunk;
    }
    receiver = producer;
    // Frozen decoders index peers[rank] + source * packet_bytes.
    for (int peer = 0; peer < kWorld; ++peer)
      receiver.peers[peer] = static_cast<unsigned char*>(rx_packets);
  }

  GemmReduceScatterArguments arguments(const void* input, const void* weight) {
    ReduceScatterArguments rs{};
    rs.reduce_scatter_num_blocks = 6;
    rs.use_barrier_queue = false; rs.use_gemmk = false; rs.per_tile_flags = false;
    rs.use_cudaMemcpyAsync = false; rs.n_split = 1; rs.sub_world_size = 8;
    rs.use_1d_ring = true; rs.use_p2p_read = true;
    GemmReduceScatterArguments args{};
    args.m = kM; args.n = kN; args.k = k; args.rank = rank;
    args.world_size = kWorld; args.nnodes = 1; args.alpha = 1.f; args.beta = 0.f;
    args.input = input; args.weight = weight;
    args.output_scatter_ptrs = outputs.data();
    args.reduce_buffer_ptrs = unused.data();
    args.barrier_ptrs = nullptr; args.avail_sms = -1;
    args.reduce_scatter_args = rs;
    return args;
  }

  OpRegistry::OpPtr make_op() {
    auto meta = make_gemm_meta(make_gemm_dtype_config(_BF16{}, _BF16{}, _Void{}, _BF16{}),
        _Sm80{}, _A100{}, _ReduceScatter{}, _RCR{}, _GemmV2{},
        make_gemm_v2_meta(_False{}), make_reduce_scatter_meta(_False{}, _IntraNode{}));
    auto rt = make_runtime_config(kM, kN, k);
    auto hparams = OpRegistry::instance().get_hparams(meta, rt);
    auto result = OpRegistry::instance().get_op(meta, hparams);
    ++initializations;
    return result;
  }
};

std::mutex handles_mutex;
std::unordered_set<Local*> handles;

Local* checked_handle(void* handle) {
  auto* value = static_cast<Local*>(handle);
  std::lock_guard<std::mutex> lock(handles_mutex);
  check(value && handles.count(value), "Unknown or destroyed LocalTaco handle");
  return value;
}

template<class Function> int guarded(Function function) {
  try {
    last_error.clear();
    function();
    return int(cudaSuccess);
  } catch (const std::exception& error) {
    last_error = error.what();
    return int(cudaErrorInvalidValue);
  } catch (...) {
    last_error = "Unknown native LocalTaco failure";
    return int(cudaErrorUnknown);
  }
}

void validate_layout(Local& local, int64_t tx_bytes, int64_t rx_bytes) {
  std::array<unsigned char, kWorld * kTiles> mask{};
  std::array<int64_t, kTiles> tx{}, rx{};
  if (local.selected)
    cuda_check(cudaMemcpyAsync(mask.data(), local.selected, mask.size(), cudaMemcpyDeviceToHost, local.stream));
  cuda_check(cudaMemcpyAsync(tx.data(), local.tx_offsets, sizeof(tx), cudaMemcpyDeviceToHost, local.stream));
  cuda_check(cudaMemcpyAsync(rx.data(), local.rx_offsets, sizeof(rx), cudaMemcpyDeviceToHost, local.stream));
  cuda_check(cudaStreamSynchronize(local.stream));
  rtx5090_local_layout::validate(local.rank, local.selected ? mask.data() : nullptr,
                                 tx.data(), rx.data(), tx_bytes, rx_bytes);
}

struct RestoreTaco {
  FluxTacoConfig previous = taco_config();
  ~RestoreTaco() { rtx5090_local_set_taco_config(previous); }
};
}  // namespace

extern "C" int rtx5090_local_protocol_version() { return 1; }
extern "C" const char* rtx5090_local_last_error() { return last_error.c_str(); }

extern "C" int rtx5090_local_create(void** handle, int rank, int k_local, int cache_submission,
    const void* selected, void* tx_bf16, void* tx_packets, void* rx_bf16, void* rx_packets,
    const void* tx_offsets, const void* rx_offsets, void* tx_wire, void* rx_wire,
    int64_t tx_bytes, int64_t rx_bytes, void* stream) {
  if (handle) *handle = nullptr;
  return guarded([&] {
    check(handle && rank >= 0 && rank < 8 && (k_local == 512 || k_local == 2048)
          && (cache_submission == 0 || cache_submission == 1), "Invalid LocalTaco exact-shape configuration");
    check(cache_submission == RTX5090_LOCAL_CONSUMER && bool(selected) == bool(RTX5090_LOCAL_SELECTIVE),
          "Native arm requires its registered selection and submission-cache policy");
    const char* gemm_only = std::getenv("FLUX_GEMM_RS_GEMM_ONLY");
    const char* local_rank = std::getenv("LOCAL_RANK");
    const char* local_world = std::getenv("LOCAL_WORLD_SIZE");
    check(gemm_only && std::strcmp(gemm_only, "1") == 0,
          "LocalTaco requires FLUX_GEMM_RS_GEMM_ONLY=1");
    check(local_rank && std::strlen(local_rank) == 1 && local_rank[0] == '0' + rank
          && local_world && std::strcmp(local_world, "8") == 0,
          "Frozen tile scheduler requires matching LOCAL_RANK and LOCAL_WORLD_SIZE=8");
    std::unique_ptr<Local> local(new Local{});
    local->rank = rank; local->k = k_local; local->cache = cache_submission;
    cuda_check(cudaGetDevice(&local->device));
    cudaDeviceProp prop{};
    cuda_check(cudaGetDeviceProperties(&prop, local->device));
    check(prop.major == 12 && prop.minor == 0 && prop.multiProcessorCount == 170,
          "LocalTaco requires an RTX5090 SM120 / 170-SM device");
    local->stream = reinterpret_cast<cudaStream_t>(stream);
    local->check_stream(stream);
    for (const void* pointer : {tx_bf16, tx_packets, rx_bf16, rx_packets, tx_wire, rx_wire})
      local_pointer(pointer, local->device);
    local_pointer(selected, local->device, true);
    local_pointer(tx_offsets, local->device); local_pointer(rx_offsets, local->device);
    check(tx_bf16 != rx_bf16 && tx_packets != rx_packets && tx_wire != rx_wire,
          "Transmit and receive buffers must have independent storage");
    local->selected = static_cast<const unsigned char*>(selected);
    local->tx_bf16 = tx_bf16; local->tx_packets = tx_packets;
    local->rx_bf16 = rx_bf16; local->rx_packets = rx_packets;
    local->tx_offsets = static_cast<const int64_t*>(tx_offsets);
    local->rx_offsets = static_cast<const int64_t*>(rx_offsets);
    local->tx_wire = tx_wire; local->rx_wire = rx_wire;
    validate_layout(*local, tx_bytes, rx_bytes);
    local->configure();
    std::lock_guard<std::mutex> lock(handles_mutex);
    handles.insert(local.get());
    *handle = local.release();
  });
}

extern "C" int rtx5090_local_reconfigure(void* handle, void* stream) {
  Local* local = nullptr;
  const int result = guarded([&] {
    local = checked_handle(handle);
    std::lock_guard<std::mutex> lock(local->mutex);
    local->check_stream(stream);
    check(local->phase == 0 && !local->cache, "Reconfigure requires an idle uncached producer");
    local->configure();
    ++local->reconfigurations;
  });
  if (result && local) local->failed = true;
  return result;
}

extern "C" int64_t rtx5090_local_reconfiguration_count(void* handle) {
  int64_t count = -1;
  const int result = guarded([&] {
    auto* local = checked_handle(handle);
    std::lock_guard<std::mutex> lock(local->mutex);
    count = local->reconfigurations;
  });
  return result ? -1 : count;
}

extern "C" int64_t rtx5090_local_workspace_bytes(void* handle) {
  int64_t count = -1;
  const int result = guarded([&] {
    auto* local = checked_handle(handle);
    std::lock_guard<std::mutex> lock(local->mutex);
    count = static_cast<int64_t>(local->workspace_bytes);
  });
  return result ? -1 : count;
}

extern "C" int rtx5090_local_produce(void* handle, const void* input, const void* weight, void* stream) {
  Local* local = nullptr;
  const int result = guarded([&] {
    local = checked_handle(handle);
    std::lock_guard<std::mutex> lock(local->mutex);
    local->check_stream(stream);
    check(local->phase == 0, "Produce cannot overwrite an unconsumed transaction");
    local_pointer(input, local->device); local_pointer(weight, local->device);
    auto args = local->arguments(input, weight);
    OpRegistry::OpPtr transient;
    if (!local->cache) transient = local->make_op();
    else if (!local->cached_op) local->cached_op = local->make_op();
    auto* op = local->cache ? local->cached_op.get() : transient.get();
    RestoreTaco restore;
    rtx5090_local_set_taco_config(local->producer);
    // _IntraNode selects plain GemmUniversalBase, not the PCIe RS operator.
    // Its argument-workspace initializer is a no-op, for cached and ordinary
    // submissions alike. CUTLASS reinitializes its GEMM scratch in run().
    check(op->get_args_workspace_size(args) == 0, "Unexpected transport argument workspace");
    const size_t bytes = op->get_workspace_size(args);
    if (bytes > local->workspace_bytes) {
      // Shape and launch parameters are immutable: allocation is first-use only.
      check(local->produces == 0 && !local->workspace, "Immutable GEMM workspace size changed");
      cuda_check(cudaMalloc(&local->workspace, bytes));
      local->workspace_bytes = bytes;
      op->initialize_args_workspace(args, local->workspace, local->stream);
    }
    op->run(args, local->workspace, local->stream);
    rtx5090_pack_tiles_kernel<<<kTiles, 256, 0, local->stream>>>(local->rank, local->selected,
        static_cast<const __nv_bfloat16*>(local->tx_bf16), static_cast<const unsigned char*>(local->tx_packets),
        local->tx_offsets, static_cast<unsigned char*>(local->tx_wire));
    cuda_check(cudaGetLastError());
    ++local->produces;
    local->phase = 1;
  });
  if (result && local) local->failed = true;
  return result;
}

extern "C" int rtx5090_local_decode(void* handle, void* output,
    const void* bias, const void* residual, void* stream) {
  Local* local = nullptr;
  const int result = guarded([&] {
    local = checked_handle(handle);
    std::lock_guard<std::mutex> lock(local->mutex);
    local->check_stream(stream);
    check(local->phase == 1, "Decode requires exactly one prior producer and NCCL exchange");
    check(bool(bias) == bool(residual), "Consumer requires both bias and residual");
    local_pointer(output, local->device);
    local_pointer(bias, local->device, true); local_pointer(residual, local->device, true);
    check(output != local->rx_bf16 && output != local->tx_bf16 && output != residual,
          "Decoder output must have independent storage");
#if !RTX5090_LOCAL_CONSUMER
    check(!bias, "This native arm has no fused bias/residual decoder");
#endif
    rtx5090_unpack_tiles_kernel<<<kTiles, 256, 0, local->stream>>>(local->rank, local->selected,
        static_cast<__nv_bfloat16*>(local->rx_bf16), static_cast<unsigned char*>(local->rx_packets),
        local->rx_offsets, static_cast<const unsigned char*>(local->rx_wire));
    cuda_check(cudaGetLastError());
#if RTX5090_LOCAL_CONSUMER
    if (bias) {
      cuda_check(static_cast<cudaError_t>(taco_ma4f_decode_reduce(local->receiver,
          local->rx_bf16, output, bias, residual, local->stream)));
      ++local->consumers;
    } else
#endif
    cuda_check(static_cast<cudaError_t>(taco_decode_reduce(local->receiver, local->rx_bf16, output, local->stream)));
    ++local->decodes;
    local->phase = 0;
  });
  if (result && local) local->failed = true;
  return result;
}

extern "C" int rtx5090_local_state(void* handle, int64_t* values, int length) {
  return guarded([&] {
    auto* local = checked_handle(handle);
    std::lock_guard<std::mutex> lock(local->mutex);
    check(values && length == 8, "Native state requires eight host int64 slots");
    const int64_t state[] = {1, local->initializations, local->produces, local->decodes,
        local->consumers, local->phase, local->failed ? 1 : 0, local->k};
    std::copy(state, state + 8, values);
  });
}

extern "C" int rtx5090_local_destroy(void* handle) {
  return guarded([&] {
    auto* local = checked_handle(handle);
    // Destruction may be called after a failed transaction, so do not use the
    // normal failed-state guard. It is never part of the measured forward.
    int current = -1;
    cuda_check(cudaGetDevice(&current));
    check(current == local->device, "Destroy LocalTaco on its creation device");
    cuda_check(cudaStreamSynchronize(local->stream));
    if (local->workspace) cuda_check(cudaFree(local->workspace));
    {
      std::lock_guard<std::mutex> lock(handles_mutex);
      handles.erase(local);
    }
    delete local;
  });
}
