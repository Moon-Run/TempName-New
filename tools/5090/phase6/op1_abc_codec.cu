// Included after the unchanged upstream codec and its basic receiver.
// Packet groups are [128 FP8 bytes, qscale, adaptive scale] or 128 BF16 values.
__global__ void op1_abc_decode_kernel(const unsigned char* packets,
    const __nv_bfloat16* local, __nv_bfloat16* output, int rank,
    int begin, int count, int raw_groups) {
  const int group = blockIdx.x, tid = threadIdx.x;
  const int idx = (begin + group) * 128 + tid;
  const size_t stride = size_t(raw_groups) * 256 + size_t(count - raw_groups) * 136;
  const size_t position = group < raw_groups ? size_t(group) * 256
      : size_t(raw_groups) * 256 + size_t(group - raw_groups) * 136;
  __shared__ float values[128];
  __nv_bfloat16 sum = __float2bfloat16(0.f);
  #pragma unroll 1
  for (int offset = 1; offset < 8; ++offset) {
    const int source = (rank + offset) % 8;
    const int slot = source - int(source > rank);
    const unsigned char* packet = packets + size_t(slot) * stride + position;
    __nv_bfloat16 decoded;
    if (group < raw_groups) {
      decoded = reinterpret_cast<const __nv_bfloat16*>(packet)[tid];
    } else {
      const float qs = *reinterpret_cast<const float*>(packet + 128);
      float adaptive = *reinterpret_cast<const float*>(packet + 132);
      __half_raw hr = __nv_cvt_fp8_to_halfraw(packet[tid], __NV_E4M3);
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
      decoded = __float2bfloat16(check_nan_inf(val) / adaptive);
    }
    sum = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(decoded));
  }
  output[idx] = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(local[idx]));
}

static bool abc_valid(int rank, int total, int begin, int count, int raw) {
  return rank >= 0 && rank < 8 && total > 0 && total <= INT_MAX / 1024 &&
      begin >= 0 && count > 0 && begin <= total - count && raw >= 0 && raw <= count;
}

extern "C" int op1_abc_encode(const void* input, void* output, int rank,
    int total, int begin, int count, int raw_groups, void* stream) {
  if (!input || !output || !abc_valid(rank, total, begin, count, raw_groups))
    return int(cudaErrorInvalidValue);
  const size_t stride = size_t(raw_groups) * 256 + size_t(count - raw_groups) * 136;
  op1_abc_encode_kernel<__nv_bfloat16, 128>
      <<<dim3(count, 7), 128, 128 * sizeof(float), (cudaStream_t)stream>>>(
      input, output, nullptr, nullptr, total * 128, size_t(total) * 256,
      stride, 0, 448.f, 1e-6f, 448.f, __NV_SATFINITE, __NV_E4M3,
      rank, begin, raw_groups);
  return int(cudaGetLastError());
}

extern "C" int op1_abc_decode(const void* packets, const void* local, void* output,
    int rank, int total, int begin, int count, int raw_groups, void* stream) {
  if (!packets || !local || !output || !abc_valid(rank, total, begin, count, raw_groups))
    return int(cudaErrorInvalidValue);
  op1_abc_decode_kernel<<<count, 128, 0, (cudaStream_t)stream>>>(
      static_cast<const unsigned char*>(packets), static_cast<const __nv_bfloat16*>(local),
      static_cast<__nv_bfloat16*>(output), rank, begin, count, raw_groups);
  return int(cudaGetLastError());
}
