// This executable is built without a GPU; running it requires the target H100s.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <set>
#include <string>
#include <vector>
#include "gemm_rs/tile_scheduler/threadblock_swizzle.hpp"
#include "phase6_host.hpp"
using Base = cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
using Swizzle = cutlass::gemm::threadblock::ThreadblockSwizzleStreamKRankOffset;
__global__ void mapping(Swizzle s, int *out, int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    auto x = s.get_tile_offset(i), b = s.Base::get_tile_offset(i);
    out[4*i] = x.m(); out[4*i+1] = x.n();
    out[4*i+2] = b.m(); out[4*i+3] = b.n();
  }
}
#define CHECK(x) do { auto e=(x); if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;} } while(0)
int main(int argc, char **argv) {
  if (argc != 6) return 64;
  int m=atoi(argv[1]), n=atoi(argv[2]), k=atoi(argv[3]), tp=atoi(argv[4]);
  if ((tp!=4 && tp!=8) || m<=0 || n<=0 || k<=0 || m%(128*tp) || n%128 || k%(32*tp)) return 65;
  int tm=m/128, tn=n/128, tiles=tm*tn;
  void *library=dlopen(argv[5],RTLD_NOW|RTLD_LOCAL);
  if(!library){fprintf(stderr,"%s\n",dlerror());return 70;}
  auto occupancy=reinterpret_cast<int(*)()>(dlsym(library,"h100_v2_occupancy"));
  if(!occupancy){fprintf(stderr,"Missing exact-kernel occupancy probe\n");return 71;}
  setenv("LOCAL_WORLD_SIZE",std::to_string(tp).c_str(),1);
  puts("rank,index,m,n,destination,base_m,base_n,sm_occupancy");
  for (int rank=0;rank<tp;++rank) {
    CHECK(cudaSetDevice(rank)); setenv("LOCAL_RANK",std::to_string(rank).c_str(),1);
    cudaDeviceProp prop; CHECK(cudaGetDeviceProperties(&prop,rank));
    if(prop.major!=9 || prop.minor!=0) return 66;
    int active=occupancy();if(active<=0)return 72;
    Swizzle sw(cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k/tp},{128,128,32},1,active,
        prop.multiProcessorCount,prop.multiProcessorCount,2,2,2,8);
    int count=sw.cohort_raster ? ((tm+Base::kCohortCtasM-1)/Base::kCohortCtasM)*
        ((tn+Base::kCohortCtasN-1)/Base::kCohortCtasN)*Base::kCtasPerCohort : tiles;
    if(count!=tiles) return 67;
    int *dev; CHECK(cudaMalloc(&dev,tiles*4*sizeof(int)));
    mapping<<<(tiles+255)/256,256>>>(sw,dev,tiles); CHECK(cudaGetLastError());
    std::vector<int> v(tiles*4); CHECK(cudaMemcpy(v.data(),dev,v.size()*sizeof(int),cudaMemcpyDeviceToHost));
    std::set<int> seen;
    for(int i=0;i<tiles;++i) {
      int x=v[4*i], y=v[4*i+1], bx=v[4*i+2], by=v[4*i+3];
      if(x<0 || x>=tm || y<0 || y>=tn || !seen.insert(x*tn+y).second) return 68;
      int expected=phase6_host_lookup(tm,tn,k/tp,rank,i);
      if(expected>=0 && x*tn+y!=expected) return 69;
      printf("%d,%d,%d,%d,%d,%d,%d,%d\n",rank,i,x,y,x/(tm/tp),bx,by,active);
    }
    CHECK(cudaFree(dev));
  }
  dlclose(library);
}
