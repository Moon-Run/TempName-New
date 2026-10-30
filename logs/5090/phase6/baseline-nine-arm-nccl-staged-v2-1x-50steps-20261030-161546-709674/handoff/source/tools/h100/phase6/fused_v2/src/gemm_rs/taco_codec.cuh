// Adapted from COCCL_TACO/src/device/compress/taco/taco.cu.
// Copyright (c) 2025, Institute of Computing Technology, Chinese Academy of Sciences.
// BSD-3-Clause; full attribution and license: tools/a800/phase4/TACO_LICENSE.txt.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include "gemm_rs/taco_runtime.h"

namespace flux_taco {
constexpr int kGroup = 128;
constexpr float kMax = 448.f;
struct Scratch {
  float exchange[kGroup];
  float warps[4];
  float ada, quant;
};

// Exactly 128 threads must participate, including padded tail coefficients.
__device__ __forceinline__ float reduce(float x, Scratch& s, bool maximum) {
  int tid = threadIdx.x;
  for (int d = 16; d; d /= 2) {
    float y = __shfl_down_sync(0xffffffff, x, d);
    x = maximum ? fmaxf(x, y) : x + y;
  }
  if ((tid & 31) == 0) s.warps[tid / 32] = x;
  __syncthreads();
  x = tid < 4 ? s.warps[tid] : 0.f;
  if (tid < 32) {
    for (int d = 16; d; d /= 2) {
      float y = __shfl_down_sync(0xffffffff, x, d);
      x = maximum ? fmaxf(x, y) : x + y;
    }
  }
  return x;  // Only thread 0 consumes this value.
}

__device__ __forceinline__ float hadamard(float x, Scratch& s) {
  int tid = threadIdx.x;
  for (int d = 1; d < 32; d *= 2) {
    float y = __shfl_xor_sync(0xffffffff, x, d);
    x = (tid & d) ? y - x : x + y;
  }
  s.exchange[tid] = x;
  __syncthreads();
  for (int d = 32; d < kGroup; d *= 2) {
    float y = s.exchange[tid ^ d];
    __syncthreads();
    x = (tid & d) ? y - x : x + y;
    s.exchange[tid] = x;
    __syncthreads();
  }
  return x * rsqrtf(float(kGroup));
}

__device__ __forceinline__ void encode(float x, int valid, unsigned char* payload,
                                      float* qs, float* as, Scratch& s) {
  float sum = reduce(x * x, s, false);
  if (threadIdx.x == 0) {
    float variance = fmaxf(sum / valid + 1e-6f, 1e-12f);
    s.ada = fminf(fmaxf(kMax * rsqrtf(variance), 1e-3f), 1e3f);
    *as = s.ada;
  }
  __syncthreads();
  x = hadamard(x * s.ada, s);
  float maximum = reduce(fabsf(x), s, true);
  if (threadIdx.x == 0) {
    s.quant = fminf(fmaxf(fmaxf(maximum, 1e-12f) / kMax, 1e-12f), 1e6f);
    *qs = s.quant;
  }
  __syncthreads();
  // Preserve upstream's nonfinite handling. The validation harness rejects
  // nonfinite inputs/outputs rather than treating this as numerical acceptance.
  x = isfinite(x) ? x : 0.f;
  float scaled = fminf(fmaxf(x / s.quant, -kMax), kMax);
  unsigned q = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
  unsigned q1 = __shfl_down_sync(0xffffffff, q, 1);
  unsigned q2 = __shfl_down_sync(0xffffffff, q, 2);
  unsigned q3 = __shfl_down_sync(0xffffffff, q, 3);
  if ((threadIdx.x & 3) == 0)
    reinterpret_cast<unsigned*>(payload)[threadIdx.x / 4] = q | (q1 << 8) | (q2 << 16) | (q3 << 24);
  __syncthreads();  // Scratch may be reused for the next row.
}

// One warp owns a complete H128 group: lane l holds coefficients l+32*j.
// Cross-warp stages of the original transform become register butterflies.
// Keep the block codec above unchanged for the independent reference path.
__device__ __forceinline__ void encode_warp(float (&x)[4], int valid,
                                           unsigned char* payload, float* qs, float* as) {
  const int lane = threadIdx.x & 31;
  float sum = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) sum += x[j] * x[j];
  #pragma unroll
  for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
  sum = __shfl_sync(0xffffffff, sum, 0);
  float ada = fminf(fmaxf(kMax * rsqrtf(fmaxf(sum / valid + 1e-6f, 1e-12f)), 1e-3f), 1e3f);
  if (lane == 0) *as = ada;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] *= ada;
    #pragma unroll
    for (int d = 1; d < 32; d *= 2) {
      float y = __shfl_xor_sync(0xffffffff, x[j], d);
      x[j] = (lane & d) ? y - x[j] : x[j] + y;
    }
  }
  // d=32, then d=64, preserving the original Hadamard signs and order.
  float a = x[0], b = x[1], c = x[2], d = x[3];
  x[0] = a + b; x[1] = a - b; x[2] = c + d; x[3] = c - d;
  a = x[0]; b = x[1]; c = x[2]; d = x[3];
  x[0] = a + c; x[1] = b + d; x[2] = a - c; x[3] = b - d;
  float maximum = 0.f;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    x[j] *= rsqrtf(float(kGroup));
    maximum = fmaxf(maximum, fabsf(x[j]));
  }
  #pragma unroll
  for (int offset = 16; offset; offset /= 2)
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffff, maximum, offset));
  maximum = __shfl_sync(0xffffffff, maximum, 0);
  float quant = fminf(fmaxf(fmaxf(maximum, 1e-12f) / kMax, 1e-12f), 1e6f);
  if (lane == 0) *qs = quant;
  #pragma unroll
  for (int j = 0; j < 4; ++j) {
    float value = isfinite(x[j]) ? x[j] : 0.f;
    float scaled = fminf(fmaxf(value / quant, -kMax), kMax);
    unsigned q = __nv_cvt_float_to_fp8(scaled, __NV_SATFINITE, __NV_E4M3);
    unsigned q1 = __shfl_down_sync(0xffffffff, q, 1);
    unsigned q2 = __shfl_down_sync(0xffffffff, q, 2);
    unsigned q3 = __shfl_down_sync(0xffffffff, q, 3);
    if ((lane & 3) == 0)
      reinterpret_cast<unsigned*>(payload)[j * 8 + lane / 4] = q | (q1 << 8) | (q2 << 16) | (q3 << 24);
  }
}

__device__ __forceinline__ float decode(const unsigned char* payload, float qs, float as,
                                       Scratch& s) {
  __half_raw raw = __nv_cvt_fp8_to_halfraw(payload[threadIdx.x], __NV_E4M3);
  float x = __half2float(__half(raw)) * qs;
  x = hadamard(x, s);
  return (isfinite(x) ? x : 0.f) / fmaxf(as, 1e-12f);
}
}  // namespace flux_taco
