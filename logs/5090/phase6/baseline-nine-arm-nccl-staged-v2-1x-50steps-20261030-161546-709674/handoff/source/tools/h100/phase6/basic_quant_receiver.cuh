// Basic receiver-only fusion. Included after the unchanged baseline codec.
// Packet math follows the pinned upstream H128 decoder, with no prefetch,
// vectorized source loads, atomics, persistent buffers or stream overlap.
// Each CTA owns one output H128 group and visits remote sources serially.
__global__ void basic_quant_decode_reduce_kernel(
    const unsigned char* packets, const __nv_bfloat16* local,
    __nv_bfloat16* output, int rank, int elements) {
  const int tid = threadIdx.x, group = blockIdx.x, idx = group * 128 + tid;
  const size_t scales = size_t(elements / 128) * sizeof(float);
  const size_t stride = size_t(elements) + 2 * scales;
  __shared__ float values[128];
  __nv_bfloat16 sum = __float2bfloat16(0.f);
  #pragma unroll 1
  for (int offset = 1; offset < 8; ++offset) {
    const int source = (rank + offset) % 8;
    const int chunk = source - int(source > rank); // NCCL ascending source order, no self.
    const unsigned char* packet = packets + size_t(chunk) * stride;
    const float qs = reinterpret_cast<const float*>(packet + elements)[group];
    float adaptive = reinterpret_cast<const float*>(packet + elements + scales)[group];
    __half_raw hr = __nv_cvt_fp8_to_halfraw(packet[idx], __NV_E4M3);
    float val = __half2float(*reinterpret_cast<__half*>(&hr)) * qs;
    #pragma unroll
    for (int s = 1; s < 32; s *= 2) {
      const float neighbor = __shfl_xor_sync(0xffffffff, val, s);
      val = (tid & s) ? neighbor - val : val + neighbor;
    }
    values[tid] = val;
    __syncthreads();
    for (int s = 32; s < 128; s *= 2) {
      const float neighbor = values[tid ^ s];
      const float next = (tid & s) ? neighbor - val : val + neighbor;
      __syncthreads();
      val = next;
      values[tid] = val;
      __syncthreads();
    }
    val *= rsqrtf(128.f);
    if (adaptive < 1e-12f) adaptive = 1e-12f;
    val = check_nan_inf(val) / adaptive;
    // Match the materialized decoder's BF16 rounding AND each PyTorch BF16 add.
    const __nv_bfloat16 decoded = __float2bfloat16(val);
    sum = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(decoded));
  }
  output[idx] = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(local[idx]));
}

extern "C" int basic_quant_decode_reduce(const void* packets, const void* local,
                                        void* output, int rank, int64_t elements,
                                        void* stream) {
  if (!packets || !local || !output || rank < 0 || rank >= 8 || !valid(7, elements))
    return int(cudaErrorInvalidValue);
  basic_quant_decode_reduce_kernel<<<elements / 128, 128, 0, (cudaStream_t)stream>>>(
      static_cast<const unsigned char*>(packets), static_cast<const __nv_bfloat16*>(local),
      static_cast<__nv_bfloat16*>(output), rank, int(elements));
  return int(cudaGetLastError());
}
