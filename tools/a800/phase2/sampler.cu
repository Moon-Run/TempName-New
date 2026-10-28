#include "sampler.h"
static thread_local Phase2Config config;
Phase2Config phase2_config(){return config;}
extern "C" __attribute__((visibility("default"))) void phase2_configure(
    void* producer,void* flags0,void* flags1,int stride,int offset,int epoch,int mode){
  config.producer=(unsigned long long*)producer;
  config.flags[0]=(int*)flags0;config.flags[1]=(int*)flags1;
  config.stride=stride;config.offset=offset;config.epoch=epoch;config.mode=mode;
}
__global__ void phase2_observe_kernel(const int* flags,const long long* indices,int count,
    unsigned long long* seen,unsigned long long* status,int epoch){
  __shared__ int done;
  if(threadIdx.x==0){done=0;status[0]=phase2_clock();}
  __syncthreads();
  unsigned long long previous=phase2_clock(),max_gap=0;
  bool first_scan=true;
  while(true){
    auto now=phase2_clock();max_gap=max(max_gap,now-previous);previous=now;
    for(int i=threadIdx.x;i<count;i+=blockDim.x){
      auto slot=indices[i];
      if(seen[slot]==0 && phase2_acquire(flags+slot)==epoch){
        auto stamp=phase2_clock();
        seen[slot]=first_scan ? (0ull-stamp) : stamp;atomicAdd(&done,1);
      }
    }
    __syncthreads();
    bool timeout=phase2_clock()-status[0]>1000000000ull;
    // Use a block-uniform exit decision; individual clocks must not diverge around barriers.
    if(threadIdx.x==0){status[2]=timeout;status[3]=done;}
    __syncthreads();
    if(status[2] || done==count)break;
    first_scan=false;
    __nanosleep(64);
  }
  status[4+threadIdx.x]=max_gap;
  if(threadIdx.x==0)status[1]=phase2_clock();
}
extern "C" __attribute__((visibility("default"))) int phase2_observe(
    void* flags,void* indices,int count,void* seen,void* status,int epoch,void* stream){
  phase2_observe_kernel<<<1,128,0,(cudaStream_t)stream>>>((int*)flags,(long long*)indices,
      count,(unsigned long long*)seen,(unsigned long long*)status,epoch);
  return (int)cudaGetLastError();
}
__global__ void phase2_timer_kernel(unsigned long long* p){
  p[0]=phase2_clock();for(int i=0;i<1000;++i)__nanosleep(1000);p[1]=phase2_clock();
}
extern "C" __attribute__((visibility("default"))) int phase2_timer_check(void* p,void* stream){
  phase2_timer_kernel<<<1,1,0,(cudaStream_t)stream>>>((unsigned long long*)p);
  return (int)cudaGetLastError();
}
__global__ void phase2_mark_kernel(unsigned long long* p,int index){p[index]=phase2_clock();}
extern "C" __attribute__((visibility("default"))) int phase2_mark(void* p,int index,void* stream){
  phase2_mark_kernel<<<1,1,0,(cudaStream_t)stream>>>((unsigned long long*)p,index);
  return (int)cudaGetLastError();
}
