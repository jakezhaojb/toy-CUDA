#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <../gputimer.h>

float* genRandArray(int length){
   float* array = (float*)malloc(length*sizeof(float));
   memset(array, 0, sizeof(float)*length);
   for(int i = 0; i < length; ++i){
      array[i] = (float)rand() / RAND_MAX;  
   }
   return array;
}


__global__ void reduce_kernel(float* d_in, float* d_out){
   int gid = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = threadIdx.x;
   for(int s = blockDim.x / 2 ; s != 0; s >>= 1){
      if (tid < s){
         d_in[gid] += d_in[gid + s];
      }
      __syncthreads();
   }
   if(tid == 0){
      d_out[blockIdx.x] = d_in[gid];   
   }
}

__global__ void reduce_kernel_shared(float* d_in, float* d_out){
   int gid = blockDim.x * blockIdx.x + threadIdx.x;
   int tid = threadIdx.x;
   extern __shared__ float s_array[];
   s_array[tid] = d_in[gid];
   __syncthreads();
   for (int s = blockDim.x / 2; s != 0; s >>= 1){
      if (tid < s) {
         s_array[tid] += s_array[tid + s] ;  
      }
      __syncthreads();
   }
   if (tid == 0){
      d_out[blockIdx.x] = s_array[0];  
   }
}

float reduce(float* array, int length){
   float *d_in, *d_out;
   cudaMalloc((void**)&d_in, sizeof(float)*length);
   cudaMalloc((void**)&d_out, 256*sizeof(float)); // TODO this could be better.
   cudaMemcpy(d_in, array, length*sizeof(float), cudaMemcpyHostToDevice);
   //reduce_kernel<<<256, 256>>>(d_in, d_out);
   reduce_kernel_shared<<<256, 256, sizeof(float)*256>>>(d_in, d_out);
   float* out = (float*)malloc(256*sizeof(float));
   cudaMemcpy(out, d_out, 256*sizeof(float), cudaMemcpyDeviceToHost);
   float res = 0;
   for (int i = 0; i < 256; ++i){
      res += out[i] ;  
   }
   return res;
}

int main(){
   GpuTimer timer;
   int length = 65536;
   float* array = genRandArray(length);
   float res_reference = 0;
   timer.Start();
   for(int i = 0; i < length; ++i){
      res_reference += array[i];
   }
   timer.Stop();
   printf("Time elapsed for serial: %g \n", timer.Elapsed());
   timer.Start();
   float res1 = reduce(array, length);
   timer.Stop();
   printf("Time elapsed for gpu: %g \n", timer.Elapsed());
   printf("%f, %f", res_reference, res1);
   return 0;
}
