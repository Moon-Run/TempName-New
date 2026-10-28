// Exhaustive rounded-boundary and deterministic random-bit comparison with CUDA.
#include <cstdio>
#include <cuda_runtime.h>
#include "gemm_rs/taco_codec.cuh"

__global__ void check(unsigned* errors) {
  unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
  unsigned u=i*747796405u+2891336453u;
  u=((u>>((u>>28u)+4u))^u)*277803737u;
  u=(u>>22u)^u;
  float x=__uint_as_float(u);
  if (isfinite(x) && fabsf(x)<=448.f &&
      flux_taco::encode_e4m3_bounded(x)!=__nv_cvt_float_to_fp8(x,__NV_SATFINITE,__NV_E4M3))
    atomicAdd(errors,1u);
  // Every positive FP32 bit pattern whose top retained bits can round to E4M3,
  // at ties and immediately on either side; include both signs and subnormals.
  if (i<4096) {
    unsigned bits=0x38800000u+(i/8)*0x80000u;
    if (bits<=__float_as_uint(448.f)) {
      int delta=int(i%4)-1;
      bits+=delta;
      if (i&4)bits|=0x80000000u;
      float v=__uint_as_float(bits);
      if (fabsf(v)<=448.f && flux_taco::encode_e4m3_bounded(v)!=__nv_cvt_float_to_fp8(v,__NV_SATFINITE,__NV_E4M3))
        atomicAdd(errors,1u);
    }
  }
}

int main() {
  unsigned *errors;
  if(cudaMallocManaged(&errors,sizeof(unsigned))!=cudaSuccess)return 2;
  *errors=0;
  check<<<16384,256>>>(errors);
  if(cudaDeviceSynchronize()!=cudaSuccess)return 3;
  unsigned count=*errors;
  cudaFree(errors);
  printf("FP8 bounded conversion mismatches: %u (4194304 random patterns plus boundary cases)\n",count);
  return count?1:0;
}
