// MA19: publish already-updated FP32 master weights to BF16 model storage.
// No Adam arithmetic, gradient accumulation, reduction, or communication here.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdint>

namespace {
constexpr int kChunk = 32768;
constexpr int kThreads = 256;
}

// Each descriptor is [FP32 source address, BF16 destination address, elements].
// Each block record is [descriptor index, first element]. Both arrays are
// immutable device allocations constructed before the timed training window.
__global__ void ma19_parameter_publish_kernel(
    const std::int64_t* descriptors, const std::int64_t* chunks) {
  const auto descriptor = chunks[2 * blockIdx.x];
  const auto first = chunks[2 * blockIdx.x + 1];
  const auto* source = reinterpret_cast<const float*>(descriptors[3 * descriptor]);
  auto* destination = reinterpret_cast<__nv_bfloat16*>(descriptors[3 * descriptor + 1]);
  const auto elements = descriptors[3 * descriptor + 2];
  const auto last = first + kChunk < elements ? first + kChunk : elements;
  // Original copy_ is vectorized. Preserve coalesced wide memory operations
  // for normal allocator-aligned parameter storage, without assuming that
  // arbitrary smoke inputs/slices are aligned or divisible by four.
  if (((reinterpret_cast<std::uintptr_t>(source) & 15) == 0) &&
      ((reinterpret_cast<std::uintptr_t>(destination) & 7) == 0)) {
    for (auto index = first + 4 * threadIdx.x; index < last; index += 4 * kThreads) {
      if (index + 3 < last) {
        const float4 values = *reinterpret_cast<const float4*>(source + index);
        const auto a = __bfloat16_as_ushort(__float2bfloat16_rn(values.x));
        const auto b = __bfloat16_as_ushort(__float2bfloat16_rn(values.y));
        const auto c = __bfloat16_as_ushort(__float2bfloat16_rn(values.z));
        const auto d = __bfloat16_as_ushort(__float2bfloat16_rn(values.w));
        const uint2 packed = make_uint2(unsigned(a) | (unsigned(b) << 16),
                                       unsigned(c) | (unsigned(d) << 16));
        *reinterpret_cast<uint2*>(destination + index) = packed;
      } else {
        for (auto tail = index; tail < last; ++tail)
          destination[tail] = __float2bfloat16_rn(source[tail]);
      }
    }
    return;
  }
  for (auto index = first + threadIdx.x; index < last; index += kThreads)
    destination[index] = __float2bfloat16_rn(source[index]);
}

extern "C" int ma19_copy_protocol() { return 19; }
extern "C" int ma19_copy_chunk_size() { return kChunk; }

extern "C" int ma19_copy_cuda(const std::int64_t* descriptors,
                              const std::int64_t* chunks, int blocks,
                              void* raw_stream) {
  if (!descriptors || !chunks || blocks <= 0)
    return int(cudaErrorInvalidValue);
  ma19_parameter_publish_kernel<<<blocks, kThreads, 0,
      reinterpret_cast<cudaStream_t>(raw_stream)>>>(descriptors, chunks);
  return int(cudaGetLastError());
}
