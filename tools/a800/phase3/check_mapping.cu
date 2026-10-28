#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <set>
#include "gemm_rs/tile_scheduler/threadblock_swizzle.hpp"
using Swizzle=cutlass::gemm::threadblock::ThreadblockSwizzleStreamKRankOffset;
__global__ void map_kernel(Swizzle swizzle,int* out){
 int idx=blockIdx.x*blockDim.x+threadIdx.x;
 if(idx<1024){auto p=swizzle.get_tile_offset(idx);out[idx*2]=p.m();out[idx*2+1]=p.n();}
}
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}}while(0)
int main(){
 setenv("LOCAL_WORLD_SIZE","2",1);
 for(int rank=0;rank<2;++rank){
  CHECK(cudaSetDevice(rank));setenv("LOCAL_RANK",rank?"1":"0",1);
  Swizzle swizzle(cutlass::gemm::GemmUniversalMode::kGemm,{4096,4096,4096},{128,128,32},1,2,108,108,2,2,2,8);
  int* ptr;CHECK(cudaMalloc(&ptr,2048*sizeof(int)));
  map_kernel<<<4,256>>>(swizzle,ptr);CHECK(cudaGetLastError());
  std::vector<int> host(2048);CHECK(cudaMemcpy(host.data(),ptr,host.size()*sizeof(int),cudaMemcpyDeviceToHost));
  std::set<int> visited;int count[2]={0,0};
  for(int i=0;i<1024;++i){int m=host[2*i],n=host[2*i+1];if(m<0||m>=32||n<0||n>=32)return 2;
   if(!visited.insert(m*32+n).second)return 3;count[m/16]++;
   printf("%d,%d,%d,%d,%d\n",rank,i,m,n,m/16);
  }
  if(count[0]!=512||count[1]!=512||visited.size()!=1024)return 4;
  CHECK(cudaFree(ptr));
 }
 fprintf(stderr,"PASS: compiled GPU mapping is bijective, 1024 tiles/rank, 512 tiles/destination\n");
}
