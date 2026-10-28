#include <cuda_runtime.h>
#include <cstdio>
__global__ void mechanism_counter_probe(float* p){int i=blockIdx.x*blockDim.x+threadIdx.x;p[i]=float(i);}
int main(){float* p;auto e=cudaMalloc(&p,4096*sizeof(float));if(e!=cudaSuccess)return 1;mechanism_counter_probe<<<16,256>>>(p);e=cudaDeviceSynchronize();cudaFree(p);return e==cudaSuccess?0:2;}
