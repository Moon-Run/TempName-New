// Included in the only generated GemmOnly registration translation unit.
// GemmDevice_0 is the actual registered operator, not a separately recreated
// scheduler. Its own argument conversion and kernel Params build the mapping.
namespace rtx5090_op1_mapping {
using Op1MappingDevice = decltype(GemmDevice_0().gemm_device());
using Op1MappingKernel = typename Op1MappingDevice::GemmKernel;
using Op1MappingSwizzle = typename Op1MappingDevice::ThreadblockSwizzle;
static_assert(std::is_same_v<Op1MappingSwizzle,
    cutlass::gemm::threadblock::Op1AInterleavedSwizzle>);

__global__ void op1_a_sorted_mapping_kernel(Op1MappingSwizzle mapping, int* output) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < 512) {
    const auto actual = mapping.get_tile_offset(index);
    const auto original = mapping.Base::get_tile_offset(index);
    output[index * 4] = actual.m();
    output[index * 4 + 1] = actual.n();
    output[index * 4 + 2] = original.m();
    output[index * 4 + 3] = original.n();
  }
}
}  // namespace rtx5090_op1_mapping

extern "C" __attribute__((visibility("default"))) int op1_a_sorted_version() { return 1; }

extern "C" __attribute__((visibility("default"))) int
op1_a_sorted_mapping(void* output, int k, void* stream) {
  using namespace rtx5090_op1_mapping;
  try {
    if (!output || (k != 512 && k != 2048)) return int(cudaErrorInvalidValue);
    int device = -1;
    auto result = cudaGetDevice(&device);
    if (result != cudaSuccess) return int(result);
    cudaDeviceProp properties{};
    result = cudaGetDeviceProperties(&properties, device);
    if (result != cudaSuccess) return int(result);
    if (properties.major != 12 || properties.minor != 0 || properties.multiProcessorCount != 170)
      return int(cudaErrorInvalidDevice);
    cudaPointerAttributes attributes{};
    result = cudaPointerGetAttributes(&attributes, output);
    if (result != cudaSuccess) return int(result);
    if (attributes.type != cudaMemoryTypeDevice || attributes.device != device)
      return int(cudaErrorInvalidValue);
    const int occupancy = Op1MappingDevice::maximum_active_blocks();
    if (occupancy <= 0) return int(cudaErrorInvalidConfiguration);
    bytedance::flux::GemmOnlyArguments arguments{
        2048, 4096, k, 1.f, 0.f, nullptr, nullptr, nullptr, nullptr};
    GemmDevice_0 op;
    const auto gemm_arguments = op.to_gemm_args(std::any(arguments), nullptr);
    const typename Op1MappingKernel::Params parameters(
        gemm_arguments, properties.multiProcessorCount, occupancy);
    if (!parameters.block_mapping.op1_target || parameters.block_mapping.op1_rank != device)
      return int(cudaErrorInvalidValue);
    op1_a_sorted_mapping_kernel<<<2, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        parameters.block_mapping, static_cast<int*>(output));
    return int(cudaGetLastError());
  } catch (...) {
    return int(cudaErrorInvalidValue);
  }
}
