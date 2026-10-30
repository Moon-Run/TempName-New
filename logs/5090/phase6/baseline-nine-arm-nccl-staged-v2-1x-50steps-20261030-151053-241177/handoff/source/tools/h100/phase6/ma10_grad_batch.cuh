// MA10: bounded BF16 -> FP32 gradient commits on the caller's original stream.
// Included only in the candidate runtime. No GEMM/communication is changed.
#ifndef H100_PHASE6_MA10_GRAD_BATCH_CUH
#define H100_PHASE6_MA10_GRAD_BATCH_CUH
#include <cuda_runtime.h>
#include <stdint.h>
#include <limits.h>

namespace ma10_gradient_batching {
constexpr int kChunkSize = 32768;
constexpr int kThreads = 256;
constexpr int kMaxTensors = 32;

// Passed by value as CUDA launch parameters; there is no device descriptor
// allocation, cudaMemcpy, scratch tensor, atomic update or auxiliary stream.
struct Batch {
  uint64_t targets[kMaxTensors];
  uint64_t sources[kMaxTensors];
  int64_t sizes[kMaxTensors];
  int first_block[kMaxTensors];
  int count;
};
static_assert(sizeof(Batch) < 4096, "MA10 launch metadata exceeds legacy ABI");

__global__ void ma10_gradient_batch_kernel(Batch batch) {
  const int block = static_cast<int>(blockIdx.x);
  int lo = 0, hi = batch.count;
  while (lo < hi) {
    const int mid = lo + (hi - lo) / 2;
    if (batch.first_block[mid] <= block) lo = mid + 1;
    else hi = mid;
  }
  const int tensor = lo - 1;
  auto* target = reinterpret_cast<float*>(batch.targets[tensor]);
  const auto* source = reinterpret_cast<const uint16_t*>(batch.sources[tensor]);
  const int64_t start = int64_t(block - batch.first_block[tensor]) * kChunkSize;
  const int64_t end = min(start + kChunkSize, batch.sizes[tensor]);
  for (int64_t i = start + threadIdx.x; i < end; i += kThreads) {
    // Bit extension preserves BF16 subnormals even under --use_fast_math.
    const float value = __uint_as_float(uint32_t(source[i]) << 16);
    float result;
    // Explicit PTX suppresses the build-wide FTZ option. This is the original
    // FP32 round-to-nearest addition after the existing BF16 GEMM rounding.
    asm("add.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(target[i]), "f"(value));
    target[i] = result;
  }
}
}  // namespace ma10_gradient_batching

extern "C" int ma10_gradient_batching_protocol() { return 10; }
extern "C" int ma10_gradient_batching_chunk_size() {
  return ma10_gradient_batching::kChunkSize;
}
extern "C" int ma10_gradient_batching_max_tensors() {
  return ma10_gradient_batching::kMaxTensors;
}

// The first three pointers point to short HOST arrays, valid until return.
// Their entries refer to live CUDA tensors retained by the Python owner.
extern "C" int ma10_gradient_batching_cuda(
    const uint64_t* targets, const uint64_t* sources, const int64_t* sizes,
    int count, void* stream) {
  using namespace ma10_gradient_batching;
  if (count < 1 || count > kMaxTensors || !targets || !sources || !sizes)
    return static_cast<int>(cudaErrorInvalidValue);
  Batch batch{};
  batch.count = count;
  int64_t blocks = 0;
  for (int i = 0; i < count; ++i) {
    if (sizes[i] <= 0 || sizes[i] > INT64_MAX - kChunkSize ||
        !targets[i] || !sources[i] || targets[i] % 4 || sources[i] % 2)
      return static_cast<int>(cudaErrorInvalidValue);
    batch.targets[i] = targets[i];
    batch.sources[i] = sources[i];
    batch.sizes[i] = sizes[i];
    batch.first_block[i] = static_cast<int>(blocks);
    blocks += (sizes[i] + kChunkSize - 1) / kChunkSize;
    if (blocks > INT_MAX) return static_cast<int>(cudaErrorInvalidValue);
  }
  ma10_gradient_batch_kernel<<<static_cast<int>(blocks), kThreads, 0,
      reinterpret_cast<cudaStream_t>(stream)>>>(batch);
  return static_cast<int>(cudaPeekAtLastError());
}
#endif  // H100_PHASE6_MA10_GRAD_BATCH_CUH
