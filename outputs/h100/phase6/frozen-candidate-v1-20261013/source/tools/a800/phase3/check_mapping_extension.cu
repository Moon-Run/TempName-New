#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <set>
#include "gemm_rs/tile_scheduler/threadblock_swizzle.hpp"
using Base=cutlass::gemm::threadblock::ThreadblockSwizzleStreamK;
using Swizzle=cutlass::gemm::threadblock::ThreadblockSwizzleStreamKRankOffset;
__global__ void map_kernel(Swizzle swizzle,int* out,int count){
 int idx=blockIdx.x*blockDim.x+threadIdx.x;
 if(idx<count){
  auto p=swizzle.get_tile_offset(idx);
  auto b=swizzle.Base::get_tile_offset(idx);
  out[idx*4]=p.m();out[idx*4+1]=p.n();out[idx*4+2]=b.m();out[idx*4+3]=b.n();
 }
}
#define CHECK(x) do{auto e=(x);if(e!=cudaSuccess){fprintf(stderr,"%s\n",cudaGetErrorString(e));return 1;}}while(0)
int main(int argc,char** argv){
 if(argc!=6)return 64;
 int m=atoi(argv[1]),n=atoi(argv[2]),k=atoi(argv[3]),tp=atoi(argv[4]),policy=atoi(argv[5]);
 if((tp!=2 && tp!=4)||policy<0||policy>2||m<=0||n<=0||k<=0||m%(tp*128)||k%tp)return 65;
 setenv("LOCAL_WORLD_SIZE",std::to_string(tp).c_str(),1);
 const int tm=m/128,tn=(n+127)/128,tiles=tm*tn;
 puts("rank,index,m,n,destination,base_m,base_n");
 for(int rank=0;rank<tp;++rank){
  CHECK(cudaSetDevice(rank));setenv("LOCAL_RANK",std::to_string(rank).c_str(),1);
  cudaDeviceProp prop;CHECK(cudaGetDeviceProperties(&prop,rank));
  // The registered 254-register, 128-thread kernel has occupancy 2 on SM80.
  Swizzle swizzle(cutlass::gemm::GemmUniversalMode::kGemm,{m,n,k/tp},{128,128,32},1,2,
                  prop.multiProcessorCount,prop.multiProcessorCount,2,2,2,8);
  const int count=swizzle.cohort_raster ? ((tm+Base::kCohortCtasM-1)/Base::kCohortCtasM)*
       ((tn+Base::kCohortCtasN-1)/Base::kCohortCtasN)*Base::kCtasPerCohort : tiles;
  // The unchanged interleaved binary uses the logical index directly. Its current
  // supported shapes must have no padded cohort slots (partial N tiles are OK).
  if(policy==2 && count!=tiles){
   fprintf(stderr,"UNSUPPORTED interleaved padded cohort: slots=%d tiles=%d\n",count,tiles);
   return 7;
  }
  int* ptr;CHECK(cudaMalloc(&ptr,count*4*sizeof(int)));
  map_kernel<<<(count+255)/256,256>>>(swizzle,ptr,count);CHECK(cudaGetLastError());
  std::vector<int> host(count*4);CHECK(cudaMemcpy(host.data(),ptr,host.size()*sizeof(int),cudaMemcpyDeviceToHost));
  std::set<int> visited;std::vector<int> owners(tp);
  for(int i=0;i<count;++i){
   int x=host[4*i],y=host[4*i+1],bx=host[4*i+2],by=host[4*i+3];
   // Padded cohort coordinates must remain padded, never become valid tiles.
   if(bx>=tm||by>=tn){if(x<tm&&y<tn)return 2;continue;}
   if(x<0||x>=tm||y<0||y>=tn)return 3;
   if(policy==2){
    int dest=(i%tp+rank)%tp,within=i/tp;
    if(x!=dest*(tm/tp)+within/tn||y!=within%tn)return 3;
   }else if(y!=by||x!=(bx+tm/tp*(rank+policy))%tm)return 3;
   if(!visited.insert(x*tn+y).second)return 4;
   int dest=x/(tm/tp);owners[dest]++;
   printf("%d,%d,%d,%d,%d,%d,%d\n",rank,i,x,y,dest,bx,by);
  }
  if(visited.size()!=tiles)return 5;
  for(auto c:owners)if(c!=tiles/tp)return 6;
  fprintf(stderr,"PASS rank=%d tiles=%d local=%d remote=%d cohort=%d sk_tiles=%d grid=%u\n",
          rank,tiles,tiles/tp,tiles-tiles/tp,swizzle.cohort_raster,swizzle.sk_tiles,swizzle.get_grid_dims().x);
  CHECK(cudaFree(ptr));
 }
}
