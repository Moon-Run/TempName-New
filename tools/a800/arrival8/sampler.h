#pragma once
#include <cuda_runtime.h>
#include <cstdint>
constexpr int MECH_TILES=2048, MECH_FRAGMENTS=16, MECH_FIELDS=8;
constexpr int MECH_SLOTS=MECH_TILES*MECH_FRAGMENTS;
struct MechConfig {
  unsigned long long* producer=nullptr;
  int* flags[8]={};
  int stride=0, offset=0, epoch=1, mode=0;
};
MechConfig mech_config();
__device__ __forceinline__ unsigned long long mech_clock(){
  unsigned long long v; asm volatile("mov.u64 %0, %%globaltimer;":"=l"(v));return v;
}
__device__ __forceinline__ void mech_publish(int* p,int v){
  asm volatile("st.release.sys.global.u32 [%0], %1;"::"l"(p),"r"(v):"memory");
}
__device__ __forceinline__ int mech_acquire(const int* p){
  int v;asm volatile("ld.acquire.sys.global.u32 %0, [%1];":"=r"(v):"l"(p):"memory");return v;
}
