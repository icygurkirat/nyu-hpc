#include <stdio.h>
#include "utils.h"
#include <math.h>

#define NUM_ITERATIONS 5000
#define NORM_THRESHOLD 1000000
#define BLOCK_SIZE 16
#define BLOCK_SIZE_1D 1024
#define index(i, j, N)  ((i)*(N)) + (j)

__global__ void jacobi_kernel(double* u, double* u_temp, double* f, int N) {
    __shared__ double smem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    smem[threadIdx.y + 1][threadIdx.x + 1] = (i < N && j < N) ? u[index(i, j, N)]: 0;
    if(threadIdx.y == 0)
        smem[0][threadIdx.x + 1] = (i>0 && i < N && j < N) ? u[index(i-1, j, N)] : 0;
    if(threadIdx.y == BLOCK_SIZE - 1)
        smem[BLOCK_SIZE + 1][threadIdx.x + 1] = (i+1 < N && j < N) ? u[index(i+1,j,N)]: 0;
    if(threadIdx.x == 0)
        smem[threadIdx.y + 1][0] = (j>0 && j < N && i < N) ? u[index(i, j-1, N)] : 0;
    if(threadIdx.x == BLOCK_SIZE - 1)
        smem[threadIdx.y + 1][BLOCK_SIZE + 1] = (j+1 < N && i < N) ? u[index(i,j+1,N)]: 0;

    __syncthreads();
    double h_rev = (N + 1) * (N + 1);
    if(i < N && j < N)
        u_temp[index(i,j,N)] = (f[index(i,j,N)]/h_rev + 
                                smem[threadIdx.y][threadIdx.x+1] + 
                                smem[threadIdx.y+1][threadIdx.x] + 
                                smem[threadIdx.y+2][threadIdx.x+1] + 
                                smem[threadIdx.y+1][threadIdx.x+2])/4.0;
}

__global__ void diff_convolution(double* u, double* u_temp, double* f, int N) {
    __shared__ double smem[BLOCK_SIZE + 2][BLOCK_SIZE + 2];
    
    int i = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int j = blockIdx.x * BLOCK_SIZE + threadIdx.x;
    smem[threadIdx.y + 1][threadIdx.x + 1] = (i < N && j < N) ? u[index(i, j, N)]: 0;
    if(threadIdx.y == 0)
        smem[0][threadIdx.x + 1] = (i>0 && i < N && j < N) ? u[index(i-1, j, N)] : 0;
    if(threadIdx.y == BLOCK_SIZE - 1)
        smem[BLOCK_SIZE + 1][threadIdx.x + 1] = (i+1 < N && j < N) ? u[index(i+1,j,N)]: 0;
    if(threadIdx.x == 0)
        smem[threadIdx.y + 1][0] = (j>0 && j < N && i < N) ? u[index(i, j-1, N)] : 0;
    if(threadIdx.x == BLOCK_SIZE - 1)
        smem[threadIdx.y + 1][BLOCK_SIZE + 1] = (j+1 < N && i < N) ? u[index(i,j+1,N)]: 0;

    __syncthreads();
    double h_rev = (N + 1) * (N + 1);
    if(i < N && j < N) {
        h_rev = (4 * smem[threadIdx.y+1][threadIdx.x+1] - 
                smem[threadIdx.y][threadIdx.x+1] - 
                smem[threadIdx.y+1][threadIdx.x] - 
                smem[threadIdx.y+2][threadIdx.x+1] - 
                smem[threadIdx.y+1][threadIdx.x+2]) * h_rev;
        h_rev = h_rev - f[index(i,j,N)];
        u_temp[index(i,j,N)] = h_rev*h_rev;
    }
}

__global__ void reduction_kernel(double* a, double* sum, long N){
  __shared__ double smem[BLOCK_SIZE_1D];
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

void jacobi(double* u, double* u_temp, double* f, int N) {
    //apply the convolution
    dim3 gridSize((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE, 1);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    jacobi_kernel<<<gridSize, blockSize>>>(u, u_temp, f, N);
}

double diffNorm(double* u, double* u_temp, double* f, double *d_sum, int N) {
    //apply the convolution
    dim3 gridSize((N+BLOCK_SIZE-1)/BLOCK_SIZE, (N+BLOCK_SIZE-1)/BLOCK_SIZE, 1);
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE, 1);
    diff_convolution<<<gridSize, blockSize>>>(u, u_temp, f, N);


    //Doing the reduction operation
    N = N*N;
    long Nb = (N+BLOCK_SIZE_1D-1)/(BLOCK_SIZE_1D);
    while (N  > 1) {
        reduction_kernel<<<Nb,BLOCK_SIZE_1D>>>(u_temp, d_sum, N);
        double *temp = u_temp;
        u_temp = d_sum;
        d_sum = temp;
        N = Nb;
        Nb = (Nb+BLOCK_SIZE-1)/(BLOCK_SIZE);
    }
    double ret;
    cudaMemcpy(&ret, u_temp, sizeof(double), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    return sqrt(ret);
}

int main(int argc, char** argv){
	int N = 1000;
	if(argc >= 2)
		N = atoi(argv[1]);

	double *u = (double*)malloc((N+2) * (N+2) * sizeof(double));
	double *f = (double*)malloc((N+2) * (N+2) * sizeof(double));

	// initialize the vectors
	for(int i = 0; i < (N+2) * (N+2); i++) {
		u[i] = 0.0;
		f[i] = 1.0;
	}

    //begin GPU calculations
    Timer t;
    t.tic();
    double *d_u, *d_u_temp, *d_f, *d_sum;
    cudaMalloc(&d_u, N * N * sizeof(double));
    cudaMalloc(&d_u_temp, N * N * sizeof(double));
    cudaMalloc(&d_f, N * N * sizeof(double));
    cudaMalloc(&d_sum, N * N * sizeof(double));
    cudaMemcpy(d_u, u, N * N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_f, f, N * N * sizeof(double), cudaMemcpyHostToDevice);
	double norm0 = diffNorm(d_u, d_u_temp, d_f, d_sum, N);
	double norm = norm0;
	printf("Residual norm after 0 iterations: %lf\n", norm0);
    printf("Beginning the simulation...\n\n");

	for(int i = 1; i <= NUM_ITERATIONS; i++) {
		jacobi(d_u, d_u_temp, d_f, N);
		double* temp = d_u_temp;
		d_u_temp = d_u;
		d_u = temp;
		norm = diffNorm(d_u, d_u_temp, d_f, d_sum, N);
		if(i%100 == 0)
			printf("Residual norm after %d iterations: %lf\n", i, norm);
		if(norm0 >= NORM_THRESHOLD * norm){
			printf("Residual norm after %d iterations: %lf\n", i, norm);
			printf("Norm is below the threshold. Exiting now\n");
			break;
		}
	}

	printf("Total time: %lfs\n", t.toc());
	printf("Reduction in residual norm: %lf\n", norm0/norm);


}
