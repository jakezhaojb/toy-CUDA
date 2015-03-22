#include <stdio.h>
#include <stdlib.h>
#include <math.h>

__global__
void histogram_kernel (float* d_in, int* d_out, int num_data, int length_bin){
   int data_idx = threadIdx.x;
   if(data_idx < num_data){
      int bin = floor(d_in[data_idx]/length_bin)+1;
      atomicAdd(&(d_out[bin]), 1);
   }
}

__global__
void hist_reduce_kernel(float* d_in, int* d_out, int num_data, int num_bin, float length_bin){
   int data_idx = threadIdx.x + blockDim.x * blockIdx.x;
   int tid = threadIdx.x;
   int out_idx = num_bin * blockIdx.x;
   extern __shared__ float s_array[];
   s_array[tid] = d_in[data_idx];
   __syncthreads();
   /*
   if (data_idx < num_data){
      int bin = floor(d_in[data_idx] / length_bin);
      atomicAdd(&(d_out[out_idx+bin]), 1);
   }*/
   if (data_idx < num_data){
      int bin = floor(s_array[tid] / length_bin) ;  
      atomicAdd(&(d_out[out_idx+bin]), 1);
   }
}

__global__
void reduce_kernel(int* d_in, int* d_out, int num_bin){
   int gid = threadIdx.x * num_bin;
   for (int s = blockDim.x / 2; s != 0; s >>= 1){
      if (gid < s * num_bin) {
         for(int i = 0; i < num_bin; ++i){
            d_in[gid+i] += d_in[gid+s*num_bin+i];  
         }
      }
      __syncthreads();
   }
   // d_out
   for (int i = 0 ; i < num_bin; ++i) {
      d_out[i] = d_in[i];
   }  
}

void histogram(float* data, int* hist, int* hist_result, int num_data, int num_bin, float length, int nblock){
   float* d_in;
   int* d_out, *d_result;
   cudaMalloc((void**)&d_in, sizeof(float)*num_data);
   cudaMalloc((void**)&d_out, nblock*sizeof(int)*num_bin);
   cudaMalloc((void**)&d_result, sizeof(int)*num_bin);
   cudaMemset(d_out, 0, sizeof(int)*num_bin*nblock);
   cudaMemset(d_out, 0, sizeof(int)*num_bin);
   cudaMemcpy(d_in, data, sizeof(float)*num_data, cudaMemcpyHostToDevice);
   hist_reduce_kernel<<<nblock,nblock,sizeof(float)*nblock>>>(d_in, d_out, num_data, num_bin, length);
   reduce_kernel<<<1, nblock, sizeof(float)*nblock>>>(d_out, d_result, num_bin);
   cudaMemcpy(hist, d_out, sizeof(int)*num_bin*nblock, cudaMemcpyDeviceToHost);
   cudaMemcpy(hist_result, d_result, sizeof(int)*num_bin, cudaMemcpyDeviceToHost);
   cudaFree(d_in);
   cudaFree(d_out);
   cudaFree(d_result);
}

float* genRandArray(int length){
   float* array = (float*)malloc(length*sizeof(float));
   memset(array, 0, sizeof(float)*length);
   for(int i = 0; i < length; ++i){
      array[i] = (float)rand() / RAND_MAX;  
   }
   return array;
}

int main(){
   int num_data = 65536;
   int blockNum = 256;
   int num_bin = 10;
   float* data = genRandArray(num_data);
   int* hist = (int*)malloc(sizeof(int)*num_bin*blockNum);
   int* hist_result = (int*)malloc(sizeof(int)*num_bin);
   memset(hist, 0, sizeof(int)*num_bin*blockNum);
   memset(hist_result, 0, sizeof(int)*num_bin);
   histogram(data, hist, hist_result, num_data, num_bin, 0.1, blockNum);
   for(int i = 0; i < num_bin; ++i){
      printf("%d ", hist_result[i]) ;  
   }
   printf("\n");
   free(hist);
   free(hist_result);
   return 0;
}
