// MA9: exact multi-tensor gradient zero counting for the frozen FP32 optimizer.
// This header is included once in the candidate Flux runtime CUDA translation
// unit. It does not alter producer, decoder, GEMM, optimizer, or NCCL arithmetic.
#ifndef H100_PHASE6_MA9_ZERO_COUNT_CUH
#define H100_PHASE6_MA9_ZERO_COUNT_CUH

#include <cuda_runtime.h>
#include <stdint.h>

namespace ma9_zero_count {

constexpr int kChunkSize = 32768;
constexpr int kThreads = 256;

// descriptors is a contiguous device int64[N,3] tensor. Each row contains
// {FP32 gradient address, element count, exclusive prefix of chunk counts}.
// Empty tensors are legal and consume zero chunks. The caller retains every
// gradient and all metadata/workspace tensors until this stream completes.
__global__ void ma9_zero_count_chunks_kernel(
    const int64_t* descriptors, int n_tensors, int64_t* partials) {
  const int chunk = static_cast<int>(blockIdx.x);
  // upper_bound(first_chunk, chunk)-1 also handles interspersed empty tensors.
  int lo = 0;
  int hi = n_tensors;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (descriptors[3 * mid + 2] <= chunk) {
      lo = mid + 1;
    } else {
      hi = mid;
    }
  }
  const int tensor = lo - 1;
  const float* values = reinterpret_cast<const float*>(
      static_cast<uintptr_t>(descriptors[3 * tensor]));
  const int64_t size = descriptors[3 * tensor + 1];
  const int64_t start =
      (static_cast<int64_t>(chunk) - descriptors[3 * tensor + 2]) * kChunkSize;
  const int64_t remaining = size - start;
  const int limit = static_cast<int>(remaining < kChunkSize ? remaining : kChunkSize);

  unsigned int zeros = 0;
  for (int i = threadIdx.x; i < limit; i += kThreads) {
    // Exactly numel-count_nonzero for FP32: both signs of zero count; NaN,
    // infinities, subnormals, and every nonzero finite value do not count.
    // Inspect bits so an embedding build's --use_fast_math/FTZ cannot turn a
    // nonzero subnormal into zero during a floating-point comparison.
    zeros += (__float_as_uint(values[start + i]) & 0x7fffffffu) == 0u;
  }
  for (int delta = 16; delta > 0; delta >>= 1) {
    zeros += __shfl_down_sync(0xffffffffu, zeros, delta);
  }
  __shared__ unsigned int warp_counts[kThreads / 32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) warp_counts[warp] = zeros;
  __syncthreads();
  if (warp == 0) {
    zeros = lane < kThreads / 32 ? warp_counts[lane] : 0;
    for (int delta = 16; delta > 0; delta >>= 1) {
      zeros += __shfl_down_sync(0xffffffffu, zeros, delta);
    }
    if (lane == 0) partials[chunk] = static_cast<int64_t>(zeros);
  }
}

__global__ void ma9_zero_count_ordered_total_kernel(
    const int64_t* descriptors, int n_tensors, int n_chunks,
    const int64_t* partials, int64_t* counts, float* total) {
  for (int tensor = threadIdx.x; tensor < n_tensors; tensor += kThreads) {
    const int64_t begin = descriptors[3 * tensor + 2];
    const int64_t end = tensor + 1 < n_tensors
        ? descriptors[3 * (tensor + 1) + 2] : n_chunks;
    int64_t zeros = 0;
    for (int64_t chunk = begin; chunk < end; ++chunk) zeros += partials[chunk];
    counts[tensor] = zeros;
  }
  // One CTA owns the final fold. This barrier also makes its global counts
  // writes visible to thread zero before it reads the original parameter order.
  __syncthreads();
  if (threadIdx.x == 0) {
    float value = 0.0f;
    for (int tensor = 0; tensor < n_tensors; ++tensor) {
      // The baseline first converts each int64 num_zeros to float32, then adds
      // it to a float32 running total. One final int64 sum-and-cast would NOT
      // preserve rounding when counts or totals exceed 2^24.
      value = __fadd_rn(__ll2float_rn(static_cast<long long>(counts[tensor])), value);
    }
    total[0] = value;
  }
}

}  // namespace ma9_zero_count

extern "C" int ma9_zero_count_protocol() { return 9; }
extern "C" int ma9_zero_count_chunk_size() { return ma9_zero_count::kChunkSize; }

// No allocation, host/device copy, collective, or stream synchronization.
// All tensor addresses are device addresses; stream is the caller's CUDA stream.
// The integer return value is a cudaError_t; the Python caller must reject errors.
extern "C" int ma9_zero_count_cuda(
    const int64_t* descriptors, int n_tensors, int n_chunks,
    int64_t* partials, int64_t* counts, float* total, void* stream) {
  if (n_tensors < 0 || n_chunks < 0 || total == nullptr ||
      (n_tensors > 0 && (descriptors == nullptr || counts == nullptr)) ||
      (n_chunks > 0 && (n_tensors == 0 || partials == nullptr))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaStream_t cuda_stream = reinterpret_cast<cudaStream_t>(stream);
  if (n_chunks > 0) {
    ma9_zero_count::ma9_zero_count_chunks_kernel<<<
        n_chunks, ma9_zero_count::kThreads, 0, cuda_stream>>>(
            descriptors, n_tensors, partials);
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) return static_cast<int>(status);
  }
  ma9_zero_count::ma9_zero_count_ordered_total_kernel<<<
      1, ma9_zero_count::kThreads, 0, cuda_stream>>>(
          descriptors, n_tensors, n_chunks, partials, counts, total);
  return static_cast<int>(cudaPeekAtLastError());
}

#endif  // H100_PHASE6_MA9_ZERO_COUNT_CUH
