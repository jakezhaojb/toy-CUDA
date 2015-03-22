#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <../gputimer.h>

__global__
void scan_kernel(float* d_in, float* d_out, int length){
   // What if more complicated cases? Across block?
   int tid = threadIdx.x;
   extern __shared__ float s_in[];
   s_in[tid] = d_in[tid];
   __syncthreads();
   for (int offset = 1; offset < length; offset <<= 1){
      if (tid >= offset)
         s_in[tid] += s_in[tid-offset];
      __syncthreads();
   }
   d_out[tid] = s_in[tid];
}


void scan(float* array, float* array_out, int length){
   float *d_in, *d_out;
   cudaMalloc((void**)&d_in, length*sizeof(float));
   cudaMalloc((void**)&d_out, length*sizeof(float));
   cudaMemset(d_out, 0, sizeof(float)*length);
   cudaMemcpy(d_in, array, length*sizeof(float), cudaMemcpyHostToDevice);
   scan_kernel<<<1, length, length*sizeof(float)>>>(d_in, d_out, length);
   cudaMemcpy(array_out, d_out, sizeof(float)*length, cudaMemcpyDeviceToHost);
   cudaFree(d_in);
   cudaFree(d_out);
}

int main(){
   float array[8] = {1,3,2,0,4,5,6,7};
   float* array_out = (float*)malloc(sizeof(float)*8);
   scan(array, array_out, 8);
   for (int i = 0; i < 8; ++i){
      printf("%f, ", array_out[i]) ;  
   }
   return 0;   
}
