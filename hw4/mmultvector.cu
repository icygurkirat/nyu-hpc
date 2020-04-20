#include <math.h>
#include <stdio.h>
#include "utils.h"

#define BLOCK_SIZE 1024

__global__ void product_kernel(double* x, double* y, long N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N)
        x[idx] = x[idx] * y[idx];
}

__global__ void reduction_kernel(double* a, double* sum, long N){
  __shared__ double smem[BLOCK_SIZE];
  int idx =blockIdx.x * blockDim.x + threadIdx.x;

  smem[threadIdx.x] = (idx < N) ? a[idx] : 0;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (threadIdx.x < s) {
		smem[threadIdx.x] += smem[threadIdx.x + s];
	}
	__syncthreads();
   }

  // write to global memory
  if (threadIdx.x == 0) sum[blockIdx.x] = smem[threadIdx.x];
}

int main() {
    Timer tt;
    long N = 10000000;

    double *h_x, *d_x, *h_y, *d_y, *d_sum;
    h_x = (double*)malloc(N * sizeof(double));
    h_y = (double*)malloc(N * sizeof(double));
    double cpu_ans = 0;
    for(int i = 0; i < N; i++) {
        h_x[i] = (drand48()-0.5) * 200;
        h_y[i] = (drand48()-0.5) * 200;
        cpu_ans += h_x[i] * h_y[i];
    }

    printf("Inner product using CPU = %f\n", cpu_ans);

    // int devices;
    // cudaGetDeviceCount(&devices);
    // cudaSetDevice(devices-1);
    
    tt.tic();
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_y, N * sizeof(double));
    cudaMalloc(&d_sum, Nb * sizeof(double));
    cudaMemcpy(d_x, h_x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, N*sizeof(double), cudaMemcpyHostToDevice);
    product_kernel<<<Nb, BLOCK_SIZE>>>(d_x, d_y, N);

    //begin reduction
    long N2 = N;
    bool flip = false;
    while (N2  > 1) {
        if(flip)
            reduction_kernel<<<Nb,BLOCK_SIZE>>>(d_sum, d_x, N2);
        else
            reduction_kernel<<<Nb,BLOCK_SIZE>>>(d_x, d_sum, N2);
        N2 = Nb;
        Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
        flip = !flip;
    }
    double gpu_ans;
    cudaMemcpy(&gpu_ans, flip?d_sum:d_sum, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double time = tt.toc();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        printf("ERROR:  %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
    printf("Inner product using GPU = %f\n", gpu_ans);
    printf("Error = %f\n", std::abs(gpu_ans - cpu_ans));
    printf("Bandwidth = %f GB/s\n", 2*N*sizeof(double) / (time)/1e9);

}