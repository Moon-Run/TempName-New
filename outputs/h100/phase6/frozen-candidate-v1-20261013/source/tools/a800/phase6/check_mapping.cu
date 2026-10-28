#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <set>
#include <string>
#include <vector>
#include "gemm_rs/tile_scheduler/threadblock_swizzle.hpp"
#include "phase6_host.hpp"
using Base=cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
using Swizzle=cutlass::gemm::threadblock::ThreadblockSwizzleStreamKRankOffset;
__global__ void mapping(Swizzle s,int* out,int count) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<count){auto x=s.get_tile_offset(i);auto b=s.Base::get_tile_offset(i);
    out[4*i]=x.m();out[4*i+1]=x.n();out[4*i+2]=b.m();out[4*i+3]=b.n();}
}
#define CHECK(x) do { auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;} } while(0)
int main(int argc,char** argv){
  if(argc!=6)return 64;
  int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),tp=atoi(argv[4]),mlp=atoi(argv[5]);
  if(tp!=4||m%(128*tp)||n%128||k%tp||m<=0||n<=0||k<=0)return 65;
  int tm=m/128,tn=n/128,tiles=tm*tn;
  setenv("LOCAL_WORLD_SIZE","4",1);puts("rank,index,m,n,destination,base_m,base_n");
  for(int rank=0;rank<4;++rank){
    CHECK(cudaSetDevice(rank));setenv("LOCAL_RANK",std::to_string(rank).c_str(),1);
    cudaDeviceProp prop;CHECK(cudaGetDeviceProperties(&prop,rank));
    Swizzle s(cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k/4},{128,128,32},1,2,
              prop.multiProcessorCount,prop.multiProcessorCount,2,2,2,8);
    int count=s.cohort_raster?((tm+Base::kCohortCtasM-1)/Base::kCohortCtasM)*
        ((tn+Base::kCohortCtasN-1)/Base::kCohortCtasN)*Base::kCtasPerCohort:tiles;
    if(count!=tiles)return 7;
    int* dev;CHECK(cudaMalloc(&dev,tiles*4*sizeof(int)));
    mapping<<<(tiles+255)/256,256>>>(s,dev,tiles);CHECK(cudaGetLastError());
    std::vector<int> data(tiles*4);CHECK(cudaMemcpy(data.data(),dev,data.size()*sizeof(int),cudaMemcpyDeviceToHost));
    std::set<int> seen;
    for(int i=0;i<tiles;++i){
      int x=data[4*i],y=data[4*i+1],bx=data[4*i+2],by=data[4*i+3];
      if(x<0||x>=tm||y<0||y>=tn)return 2;
      int expected=phase6_host_lookup(tm,tn,k/4,rank,i);
      if(mlp){if(expected<0||x*tn+y!=expected)return 3;}
      else if(expected>=0||x!=(bx+tm/4*rank)%tm||y!=by)return 4;
      if(!seen.insert(x*tn+y).second)return 5;
      printf("%d,%d,%d,%d,%d,%d,%d\n",rank,i,x,y,x/(tm/4),bx,by);
    }
    if(seen.size()!=tiles)return 6;
    CHECK(cudaFree(dev));fprintf(stderr,"PASS rank=%d tiles=%d mlp=%d\n",rank,tiles,mlp);
  }
}
