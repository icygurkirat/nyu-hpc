#include <stdio.h>
#include "utils.h"
#include <math.h>
#include <omp.h>

#define NUM_ITERATIONS 5000
#define NORM_THRESHOLD 1000000
#define index(i, j, N)  ((i)*(N+2)) + (j)

// runs one iteration of jacobi
void jacobi(double* u, double* u_temp, double* f, int N) {
	double h_rev = (N + 1) * (N + 1);

    #pragma omp parallel for
    for(int i = 1; i <= N; i++) {
        for(int j = 1; j <= N; j++) {
            u_temp[index(i,j,N)] = (f[index(i,j,N)]/h_rev + u[index(i-1,j,N)] + u[index(i+1,j,N)] + u[index(i,j-1,N)] + u[index(i,j+1,N)])/4.0;
        }
    }
}

// returns the residual norm ||Au-f||
double diffNorm(double* u, double* f, int N) {
    double norm = 0;
    double h_rev = (N + 1) * (N + 1);

    #pragma omp parallel for reduction(+:norm)
    for(int i = 1; i <= N; i++) {
        for(int j = 1; j <= N; j++) {
            double value = (4 * u[index(i,j,N)] - u[index(i-1,j,N)] - u[index(i,j-1,N)] - u[index(i+1,j,N)] - u[index(i,j+1,N)]) * h_rev;
            norm += pow(value - f[index(i,j,N)], 2);
        }
    }

	return sqrt(norm);

}

int main(int argc, char** argv){
	int N = 10;
	if(argc >= 2)
		N = atoi(argv[1]);
    
    if(argc >= 3) {
        int numThreads = atoi(argv[2]);
        omp_set_num_threads(numThreads);
    }

	double *u = (double*)malloc((N+2) * (N+2) * sizeof(double));
	double *u_temp = (double*)malloc((N+2) * (N+2) * sizeof(double));	// to store temporary vaues when running iterations
	double *f = (double*)malloc((N+2) * (N+2) * sizeof(double));

	// initialize the vectors
	for(int i = 0; i < (N+2) * (N+2); i++) {
		u[i] = 0.0;
		u_temp[i] = 0.0;
		f[i] = 1.0;
	}

	double norm0 = diffNorm(u, f, N);
	double norm = norm0;
	printf("Residual norm after 0 iterations: %lf\n", norm0);
    printf("Beginning the simulation...\n\n");
	Timer t; t.tic();

	for(int i = 1; i <= NUM_ITERATIONS; i++) {
		jacobi(u, u_temp, f, N);
		double* temp = u_temp;
		u_temp = u;
		u = temp;
		norm = diffNorm(u, f, N);
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