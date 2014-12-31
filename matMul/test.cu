#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../gputimer.h"

__global__ void matrixMulKernal(float *a, float* b, float* c, int width) {
   int row = blockDim.y * blockIdx.y + threadIdx.y;
   int col = blockDim.x * blockIdx.x + threadIdx.x;
   int tmp = 0;
   if ((row < width) && (col < width)) {
      for (int i = 0; i < width; ++i) {
         tmp += a[row*width+i] * b[i*width+col];
      }
      c[row*width+col] = tmp;
   }
}

void matrixMul(float* a, float* b, float* c, int width){
   int size = width * width * sizeof(float);
   float *d_a, *d_b, *d_c;
   cudaMalloc((void**)&d_a, sizeof(float)*width*width);
   cudaMalloc((void**)&d_b, sizeof(float)*width*width);
   cudaMalloc((void**)&d_c, sizeof(float)*width*width);
   cudaMemcpy(d_a, a, size, cudaMemcpyHostToDevice);
   cudaMemcpy(d_b, b, size, cudaMemcpyHostToDevice);
   matrixMulKernal<<<dim3(40,60), dim3(3,3)>>>(d_a, d_b, d_c, width);
   cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);
   cudaFree(d_a);
   cudaFree(d_b);
   cudaFree(d_c);
}

int main(int argc, const char *argv[])
{
   GpuTimer timer;
   float a[9] = {1,2,3,4,5,6,7,8,9};
   float b[9] = {5,6,7,8,4,2,24,6,7};
   float *c = (float*)malloc(sizeof(float)*9);
   memset(c, 0, sizeof(float)*9);
   timer.Start();
   matrixMul(a, b, c, 3);
   timer.Stop();
   int i, j;
   for (i = 0; i < 3; i++) {
      for (j = 0; j < 3; j++) {
         printf("%g ", c[i*3+j]);         
      }
      printf("\n");
   }
   printf("Time elapsed:%g ms\n", timer.Elapsed());
   free(c);
   return 0;
}
