#include <algorithm>
#include <stdio.h>
#include <math.h>
#include <omp.h>

// Scan A array and write result into prefix_sum array;
// use long data type to avoid overflow
void scan_seq(long* prefix_sum, const long* A, long n) {
  if (n == 0) return;
  prefix_sum[0] = 0;
  for (long i = 1; i < n; i++) {
    prefix_sum[i] = prefix_sum[i-1] + A[i-1];
  }
}

void scan_omp(long* prefix_sum, const long* A, long n) {
  #pragma omp parallel
  {
    int tid = omp_get_thread_num(), p = omp_get_num_threads(), len = n/p;
    
    //indices for local section
    int i = tid * len, j = (tid == p-1)?(n-1):(i+len-1);

    //find local sum
    long sum = (i==0?0:A[i-1]);
    for(int ii = i+1; ii <= j; ii++)
      sum += A[ii-1];
    
    //store this sum in global array
    prefix_sum[i] = sum;

    #pragma omp barrier

    sum = 0;
    for(int ii = 0; ii < tid; ii++)
      sum += prefix_sum[ii*len];

    #pragma omp barrier

    //add this additional sum
    prefix_sum[i] = sum + (i==0?0:A[i-1]);
    for(int ii = i+1; ii <= j; ii++)
      prefix_sum[ii] = prefix_sum[ii-1] + A[ii-1];

  }
}

int main(int argc, char** argv) {
  #ifdef _OPENMP
    if(argc >= 2) {
        int numThreads = atoi(argv[1]);
        omp_set_num_threads(numThreads);
    }
	#endif

  long N = 100000000;
  long* A = (long*) malloc(N * sizeof(long));
  long* B0 = (long*) malloc(N * sizeof(long));
  long* B1 = (long*) malloc(N * sizeof(long));
  for (long i = 0; i < N; i++) A[i] = rand();

  double tt = omp_get_wtime();
  scan_seq(B0, A, N);
  printf("sequential-scan = %fs\n", omp_get_wtime() - tt);

  tt = omp_get_wtime();
  scan_omp(B1, A, N);
  printf("parallel-scan   = %fs\n", omp_get_wtime() - tt);

  long err = 0;
  for (long i = 0; i < N; i++) err = std::max(err, std::abs(B0[i] - B1[i]));
  printf("error = %ld\n", err);

  free(A);
  free(B0);
  free(B1);
  return 0;
}
