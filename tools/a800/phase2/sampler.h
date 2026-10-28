#pragma once
#include <cuda_runtime.h>
#include <cstdint>
constexpr int P2_TILES=2048, P2_FRAGMENTS=16, P2_FIELDS=8;
constexpr int P2_SLOTS=P2_TILES*P2_FRAGMENTS;
struct Phase2Config {
  unsigned long long* producer=nullptr;
  int* flags[2]={nullptr,nullptr};
  int stride=0, offset=0, epoch=1, mode=0;
};
Phase2Config phase2_config();
__device__ __forceinline__ unsigned long long phase2_clock(){
  unsigned long long v; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));return v;
}
__device__ __forceinline__ void phase2_publish(int* p,int v){
  asm volatile("st.release.sys.global.u32 [%0], %1;"::"l"(p),"r"(v):"memory");
}
__device__ __forceinline__ int phase2_acquire(const int* p){
  int v;asm volatile("ld.acquire.sys.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;
}
