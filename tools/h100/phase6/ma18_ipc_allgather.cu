// MA18 fixed-size IPC AllGather transport. No new GPU arithmetic or kernels.
#include <cuda_runtime_api.h>
#include <cstddef>
#include <cstdint>

namespace {
constexpr std::size_t kBytes = 2 * 1024 * 1024;
constexpr int kWorld = 8;
}

extern "C" int ma18_ipc_protocol() {
// The existing Flux Python GroupBarrier must resolve to the original CUDA IPC
// binary CAS implementation, not an NVSHMEM implementation.
#ifdef FLUX_SHM_USE_NVSHMEM
  return 0;
#else
  return 18;
#endif
}

extern "C" int ma18_ipc_stage_cuda(
    void *staging, const void *input, std::uint64_t bytes, void *raw_stream) {
  if (!staging || !input || staging == input || bytes != kBytes)
    return int(cudaErrorInvalidValue);
  return int(cudaMemcpyAsync(staging, input, kBytes, cudaMemcpyDeviceToDevice,
                             reinterpret_cast<cudaStream_t>(raw_stream)));
}

extern "C" int ma18_ipc_pull_cuda(
    void *output, const void *const *peers, int rank, std::uint64_t bytes,
    void *raw_stream) {
  if (!output || !peers || rank < 0 || rank >= kWorld || bytes != kBytes)
    return int(cudaErrorInvalidValue);
  for (int source = 0; source < kWorld; ++source)
    if (!peers[source] || peers[source] == output)
      return int(cudaErrorInvalidValue);
  auto stream = reinterpret_cast<cudaStream_t>(raw_stream);
  auto *destination = static_cast<char *>(output);
  // Rotate source order so simultaneous receivers do not all pull source zero
  // first. The destination remains the exact communicator-rank layout.
  for (int offset = 0; offset < kWorld; ++offset) {
    const int source = (rank + offset) % kWorld;
    const auto status = cudaMemcpyAsync(destination + source * kBytes,
        peers[source], kBytes, cudaMemcpyDefault, stream);
    if (status != cudaSuccess)
      return int(status);
  }
  return int(cudaSuccess);
}
