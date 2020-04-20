#include <math.h>
#include <stdio.h>
#include "utils.h"

#define BLOCK_SIZE 128
#define BLOCK_SIZE_2D 32
#define index(i, j, N)  ((i)*(N)) + (j)

__global__ void product_kernel(double* d_mat, double* d_x, long N) {
    __shared__ double smem[BLOCK_SIZE_2D];
    int i = blockIdx.x * BLOCK_SIZE_2D + threadIdx.x, j = blockIdx.y * BLOCK_SIZE_2D + threadIdx.y;
    if(threadIdx.x == 0 && j < N)
        smem[threadIdx.y] = d_x[j];
    __syncthreads();

    if(i < N && j < N)
        d_mat[index(i,j,N)] = d_mat[index(i,j,N)] * smem[threadIdx.y];
}

__global__ void reduction_kernel(double* a, double* sum, long N, long N_total){
  __shared__ double smem[BLOCK_SIZE];
  int i = blockIdx.x, j = blockIdx.y * blockDim.x + threadIdx.x;

  smem[threadIdx.x] = (j < N) ? a[index(i,j,N_total)] : 0;
  __syncthreads();

  for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
	if (threadIdx.x < s) {
		smem[threadIdx.x] += smem[threadIdx.x + s];
	}
	__syncthreads();
   }

  // write to global memory
  if (threadIdx.x == 0) sum[index(blockIdx.x, blockIdx.y, N_total)] = smem[threadIdx.x];
}

int main() {
    Timer tt;
    long N = 10000;

    // mat*x = y
    double *h_mat, *d_mat, *h_x, *d_x, *d_sum;
    h_mat = (double*)malloc(N * N * sizeof(double));
    h_x = (double*)malloc(N * N * sizeof(double));
    for(int i = 0; i < N; i++) {
        h_x[i] = (drand48()-0.5) * 200;
        for(int j = 0; j < N; j++)
            h_mat[index(i,j,N)] = (drand48()-0.5) * 200;
    }

    //calculating in CPU
    double *cpu_ans = (double*)malloc(N * sizeof(double));
    for(int i = 0; i < N; i++) {
        cpu_ans[i] = 0;
        for(int j = 0; j < N; j++)
            cpu_ans[i] += h_mat[index(i,j,N)] * h_x[j];
    }
    printf("CPU done\n");

    // int devices;
    // cudaGetDeviceCount(&devices);
    // cudaSetDevice(devices-1);

    //now calculating in GPU
    
    tt.tic();
    cudaMalloc(&d_mat, N * N * sizeof(double));
    cudaMalloc(&d_x, N * sizeof(double));
    cudaMalloc(&d_sum, N * N * sizeof(double));
    cudaMemcpy(d_x, h_x, N*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mat, h_mat, N*N*sizeof(double), cudaMemcpyHostToDevice);
    dim3 gridSize((N+BLOCK_SIZE_2D-1)/(BLOCK_SIZE_2D), (N+BLOCK_SIZE_2D-1)/(BLOCK_SIZE_2D), 1);
    dim3 blockSize(BLOCK_SIZE_2D, BLOCK_SIZE_2D, 1);
    product_kernel<<<gridSize, blockSize>>>(d_mat, d_x, N);

    //begin reduction
    gridSize.x = N;
    blockSize.y = 1;
    blockSize.x = BLOCK_SIZE;
    long N2 = N;
    long Nb = (N+BLOCK_SIZE-1)/(BLOCK_SIZE);
    while (N2  > 1) {
        gridSize.y = Nb;

        reduction_kernel<<<gridSize,blockSize>>>(d_mat, d_sum, N2, N);

        double *temp = d_sum;
        d_sum = d_mat;
        d_mat = temp;

        N2 = Nb;
        Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    }
    double *gpu_ans = (double*)malloc(N * N * sizeof(double));
    cudaMemcpy(gpu_ans, d_mat, N*N*sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    double time = tt.toc();
    cudaError_t error = cudaGetLastError();
    if(error!=cudaSuccess) {
        printf("ERROR:  %s\n", cudaGetErrorString(error) );
        exit(-1);
    }
    printf("Bandwidth = %f GB/s\n", (2*N*N + N)*sizeof(double) / (time)/1e9);


    //calculating difference from CPU
    double maxDiff = 0;
    for(int i = 0; i < N; i++)
        maxDiff = std::max(maxDiff, std::abs(gpu_ans[index(i,0,N)] - cpu_ans[i]));

    printf("Max Error = %f\n", maxDiff);

}