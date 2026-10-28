#include "sampler.h"
static thread_local MechConfig config;
MechConfig mech_config(){return config;}
extern "C" __attribute__((visibility("default"))) int mech_configure(
    void* producer,void** flags,int world,int stride,int offset,int epoch,int mode){
  if((world!=2 && world!=4 && world!=8) || stride<0 || offset<0 || mode<0 || mode>2)return 1;
  config=MechConfig{};
  config.producer=(unsigned long long*)producer;
  for(int i=0;i<world;++i)config.flags[i]=(int*)flags[i];
  config.stride=stride;config.offset=offset;config.epoch=epoch;config.mode=mode;
  return 0;
}
__global__ void mech_observe_kernel(const int* flags,const long long* indices,int count,
    unsigned long long* seen,unsigned long long* status,int epoch){
  __shared__ int done;
  if(threadIdx.x==0){done=0;status[0]=mech_clock();}
  __syncthreads();
  unsigned long long previous=mech_clock(),max_gap=0;
  bool first_scan=true;
  while(true){
    auto now=mech_clock();max_gap=max(max_gap,now-previous);previous=now;
    for(int i=threadIdx.x;i<count;i+=blockDim.x){
      auto slot=indices[i];
      if(seen[slot]==0 && mech_acquire(flags+slot)==epoch){
        auto stamp=mech_clock();
        seen[slot]=first_scan ? (0ull-stamp) : stamp;atomicAdd(&done,1);
      }
    }
    __syncthreads();
    bool timeout=mech_clock()-status[0]>1000000000ull;
    // Use a block-uniform exit decision; individual clocks must not diverge around barriers.
    if(threadIdx.x==0){status[2]=timeout;status[3]=done;}
    __syncthreads();
    if(status[2] || done==count)break;
    first_scan=false;
    __nanosleep(64);
  }
  status[4+threadIdx.x]=max_gap;
  if(threadIdx.x==0)status[1]=mech_clock();
}
extern "C" __attribute__((visibility("default"))) int mech_observe(
    void* flags,void* indices,int count,void* seen,void* status,int epoch,void* stream){
  mech_observe_kernel<<<1,128,0,(cudaStream_t)stream>>>((int*)flags,(long long*)indices,
      count,(unsigned long long*)seen,(unsigned long long*)status,epoch);
  return (int)cudaGetLastError();
}
__global__ void mech_timer_kernel(unsigned long long* p){
  p[0]=mech_clock();for(int i=0;i<1000;++i)__nanosleep(1000);p[1]=mech_clock();
}
extern "C" __attribute__((visibility("default"))) int mech_timer_check(void* p,void* stream){
  mech_timer_kernel<<<1,1,0,(cudaStream_t)stream>>>((unsigned long long*)p);
  return (int)cudaGetLastError();
}
__global__ void mech_mark_kernel(unsigned long long* p,int index){p[index]=mech_clock();}
extern "C" __attribute__((visibility("default"))) int mech_mark(void* p,int index,void* stream){
  mech_mark_kernel<<<1,1,0,(cudaStream_t)stream>>>((unsigned long long*)p,index);
  return (int)cudaGetLastError();
}
