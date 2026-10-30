// MAV2 additions to the pinned TACO codec. A group is always FP8 on the wire.
// qscale is unchanged; adaptive == -1 marks direct scaled FP8 (D8).
#include <cassert>
#include <cfloat>

namespace mav2 {
constexpr float kRelativeL2 = 0.025f;
constexpr float kMaxAbs = 0.0625f;
constexpr float kDirectTag = -1.f;

__device__ __forceinline__ float block_max(float x, float* scratch) {
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  x = warpReduceMax(x);
  if (lane == 0) scratch[warp] = x;
  __syncthreads();
  x = tid < 4 ? scratch[tid] : 0.f;
  if (warp == 0) x = warpReduceMax(x);
  if (tid == 0) scratch[0] = x;
  __syncthreads();
  x = scratch[0];
  __syncthreads();
  return x;
}

__device__ __forceinline__ float block_sum(float x, float* scratch) {
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
  x = warpReduceSum(x);
  if (lane == 0) scratch[warp] = x;
  __syncthreads();
  x = tid < 4 ? scratch[tid] : 0.f;
  if (warp == 0) x = warpReduceSum(x);
  if (tid == 0) scratch[0] = x;
  __syncthreads();
  x = scratch[0];
  __syncthreads();
  return x;
}

// Returns uniformly across the CTA. mode -1 = select, 0 = H8, 1 = D8.
// mode 1 is also the formal D8 arm: it performs no quality selection or fallback.
__device__ __forceinline__ bool try_direct(float x, unsigned char* out,
    float* qscales, float* tags, int idx, int group, int mode, float* scratch) {
  assert(isfinite(x)); // Never inherit upstream check_nan_inf's silent clearing.
  if (mode == 0) return false;
  const float amax = block_max(fabsf(x), scratch);
  // BF16's smallest nonzero value / 448 remains representable in FP32.
  // Explicit correctly rounded division avoids reciprocal overflow for tiny s.
  const float scale = amax == 0.f ? 1.f : fmaxf(__fdiv_rn(amax, 448.f), 0x1p-149f);
  const float scaled = amax == 0.f ? 0.f : __fdiv_rn(x, scale);
  const unsigned char q = __nv_cvt_float_to_fp8(clamp_fp8(scaled), __NV_SATFINITE, __NV_E4M3);
  if (mode == 1) {
    out[idx] = q;
    if (threadIdx.x == 0) { qscales[group] = scale; tags[group] = kDirectTag; }
    return true; // No roundtrip, quality reductions, adaptive scale or Hadamard.
  }
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(q, __NV_E4M3);
  const float decoded = __bfloat162float(__float2bfloat16(
      __half2float(*reinterpret_cast<const __half*>(&hr)) * scale));
  const float normalized = amax == 0.f ? 0.f : __fdiv_rn(x, amax);
  const float restored = amax == 0.f ? 0.f : __fdiv_rn(decoded, amax);
  const float error = restored - normalized;
  const float energy = block_sum(normalized * normalized, scratch);
  const float mse = block_sum(error * error, scratch);
  const float max_error = block_max(fabsf(error), scratch);
  const bool direct = mse <= kRelativeL2 * kRelativeL2 * energy && max_error <= kMaxAbs;
  if (direct) {
    out[idx] = q;
    if (threadIdx.x == 0) { qscales[group] = scale; tags[group] = kDirectTag; }
  }
  return direct;
}

__device__ __forceinline__ float decode(const unsigned char* packet,
    int elements, int group, int idx, float* values) {
  const int tid = threadIdx.x;
  const size_t scales = size_t(elements / 128) * sizeof(float);
  const float qs = reinterpret_cast<const float*>(packet + elements)[group];
  float adaptive = reinterpret_cast<const float*>(packet + elements + scales)[group];
  assert(isfinite(qs) && qs > 0.f && isfinite(adaptive));
  assert(adaptive == kDirectTag || adaptive > 0.f);
  const __half_raw hr = __nv_cvt_fp8_to_halfraw(packet[idx], __NV_E4M3);
  float val = __half2float(*reinterpret_cast<const __half*>(&hr)) * qs;
  if (adaptive == kDirectTag) return val;
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
    val = next; values[tid] = val;
    __syncthreads();
  }
  val *= rsqrtf(128.f);
  if (adaptive < 1e-12f) adaptive = 1e-12f;
  return check_nan_inf(val) / adaptive;
}
} // namespace mav2

// The H8 branch below is generated from the pinned upstream encoder verbatim;
// only its name, an extra mode argument and a D8 prefix are changed.
#include "mav2_encoder.cuh"

__global__ void mav2_decode_kernel(const unsigned char* packets, __nv_bfloat16* output,
                                   int elements) {
  const int group = blockIdx.x, idx = group * 128 + threadIdx.x;
  const size_t stride = size_t(elements / 128) * 136;
  __shared__ float values[128];
  output[size_t(blockIdx.y) * elements + idx] = __float2bfloat16(
      mav2::decode(packets + size_t(blockIdx.y) * stride, elements, group, idx, values));
}

__global__ void mav2_decode_reduce_kernel(const unsigned char* packets,
    const __nv_bfloat16* local, __nv_bfloat16* output, int rank, int elements) {
  const int group = blockIdx.x, idx = group * 128 + threadIdx.x;
  const size_t stride = size_t(elements / 128) * 136;
  __shared__ float values[128];
  __nv_bfloat16 sum = __float2bfloat16(0.f);
  #pragma unroll 1
  for (int offset = 1; offset < 8; ++offset) {
    const int source = (rank + offset) % 8;
    const int chunk = source - int(source > rank);
    const __nv_bfloat16 decoded = __float2bfloat16(mav2::decode(
        packets + size_t(chunk) * stride, elements, group, idx, values));
    sum = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(decoded));
    __syncthreads();
  }
  output[idx] = __float2bfloat16(__bfloat162float(sum) + __bfloat162float(local[idx]));
}

extern "C" int mav2_encode(const void* input, void* output, int chunks,
                            int64_t elements, int mode, void* stream) {
  if (!input || !output || !valid(chunks, elements) || mode < -1 || mode > 1)
    return int(cudaErrorInvalidValue);
  const size_t scales = size_t(elements / 128) * sizeof(float);
  mav2_encode_kernel<__nv_bfloat16, 128>
      <<<dim3(elements / 128, chunks), 128, 128 * sizeof(float), (cudaStream_t)stream>>>(
          input, output, nullptr, nullptr, int(elements), size_t(elements) * 2,
          size_t(elements) + 2 * scales, scales, 448.f, 1e-6f, 448.f,
          __NV_SATFINITE, __NV_E4M3, mode);
  return int(cudaGetLastError());
}

extern "C" int mav2_decode(const void* input, void* output, int chunks,
                            int64_t elements, void* stream) {
  if (!input || !output || !valid(chunks, elements)) return int(cudaErrorInvalidValue);
  mav2_decode_kernel<<<dim3(elements / 128, chunks), 128, 0, (cudaStream_t)stream>>>(
      static_cast<const unsigned char*>(input), static_cast<__nv_bfloat16*>(output), int(elements));
  return int(cudaGetLastError());
}

extern "C" int mav2_decode_reduce(const void* packets, const void* local,
    void* output, int rank, int64_t elements, void* stream) {
  if (!packets || !local || !output || rank < 0 || rank >= 8 || !valid(7, elements))
    return int(cudaErrorInvalidValue);
  mav2_decode_reduce_kernel<<<elements / 128, 128, 0, (cudaStream_t)stream>>>(
      static_cast<const unsigned char*>(packets), static_cast<const __nv_bfloat16*>(local),
      static_cast<__nv_bfloat16*>(output), rank, int(elements));
  return int(cudaGetLastError());
}
