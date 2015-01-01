#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "../gputimer.h"

void generateArray(float *obj, const int length){
   memset(obj, 0, sizeof(float) * length);
   for (int i = 0; i < length; i++) {
      obj[i] = (float)rand() / RAND_MAX;
   }
}

__global__
void global_reduce_kernel(float *d_in, float *d_out){
   int gid = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = threadIdx.x;
   for(int s = blockDim.x / 2; s > 0; s >>= 1){
      if (tid < s) {
         d_in[gid] += d_in[gid + s];     
      }     
      __syncthreads();
   }
   if(tid == 0){
      d_out[blockIdx.x] = d_in[gid];  
   }
}


__global__
void share_reduce_kernel(float *d_in, float *d_out){
   int gid = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = threadIdx.x;
   extern __shared__ float s_array[];
   s_array[tid] = d_in[gid];
   __syncthreads();
   for (int s = blockDim.x/2; s > 0; s >>= 1){
      if (tid < s){
         s_array[tid] += s_array[tid + s];
      }
      __syncthreads();
   }
   if(tid == 0){
      d_out[blockIdx.x] = s_array[0];   
   }
   __syncthreads();
}


float reduce(float* array, int length, bool kernelType){
   int length_ = int(sqrt(length));
   int size = length * sizeof(float);
   int size_ = length_ * sizeof(float);
   float *d_in, *d_out;
   cudaMalloc((void**)&d_in, size);
   cudaMalloc((void**)&d_out, size_);
   cudaMemcpy(d_in, array, size, cudaMemcpyHostToDevice);
   if (kernelType)
      global_reduce_kernel<<<length_, length_>>>(d_in, d_out);
   else{
      share_reduce_kernel<<<length_, length_, sizeof(float)*length_>>>(d_in, d_out);
   }
   float *out = (float*)malloc(length_ * sizeof(float));
   cudaMemcpy(out, d_out, size_, cudaMemcpyDeviceToHost);
   float result = 0;
   for (int i = 0; i < length_; i++){
      result += out[i];
   }
   return result;
}


int main(int argc, const char *argv[])
{
   GpuTimer timer;
   srand(time(NULL));
   float* randArray = (float*)malloc(sizeof(float)*65536);
   generateArray(randArray, 65536);
   float refRes = 0;
   for (int i = 0; i < 65536; i++) {
      refRes += randArray[i];
   }
   printf("referece: %g\n", refRes);
   timer.Start();
   float res = reduce(randArray, 65536, 1);
   timer.Stop();
   printf("global kernal gpuCalu: %g\n", res);
   printf("Time elapsed for global kernel: %g \n", timer.Elapsed());
   
   printf("------------------------------------------------\n");
   // shared kernel
   timer.Start();
   float res2 = reduce(randArray, 65536, 0);
   timer.Stop();
   printf("shared kernel gpuCalu: %g\n", res2);
   printf("Time elapsed for shared kernel: %g \n", timer.Elapsed());
   return 0;
}
