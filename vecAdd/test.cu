#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "gputimer.h"

#define checkCudaError(err) { __checkCudaError((err), __FILE__, __LINE__); }

void __checkCudaError(cudaError_t err, const char* file, int line){
    if(err != cudaSuccess){
        fprintf(stderr, "%s(%i), CUDA RuntimeError %d, %s\n", file, line, int(err), cudaGetErrorString(err));    
        exit(-1);
    }        
}


__global__ void vecAddKernel(float* A, float* B, float* C, int n){
    //int i = threadIdx.x + blockDim.x * blockIdx.x;
    int i = threadIdx.x;
    if(i < n){
        C[i] = A[i] + B[i];
    }
}


void vecAdd(float* A, float* B, float* C, int n){
    int size = n * sizeof(float);
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);
    vecAddKernel<<<1, 256>>>(d_A, d_B, d_C, n);
    checkCudaError( cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost) );
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, const char *argv[])
{
    GpuTimer timer;
    float A[10] = {1,2,3,4,5,6,7,8,9,10};
    float B[10] = {10,20,30,40,50,60,70,80,90,100};
    //float* C = malloc(10 * sizeof(float));
    float C[10] = {0};
    timer.Start();
    vecAdd(A, B, C, 10);
    timer.Stop();
    for (int i = 0; i < 10; i++) {
        printf("%f, ", C[i]);
    }
    printf("\n");
    printf("Time elapsed: %g ms \n", timer.Elapsed());
    return 0;
}
