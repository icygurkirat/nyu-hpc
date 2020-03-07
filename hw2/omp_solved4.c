/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

// Segmentation fault occurs because the size of array (declared statically) will exceed 8MB which might exceed the usual limit of stack memory.
// The chance of stack-overflow increases considerably when multiple threads are spawned, because each thread will have its own stack space which will contain this huge array.
//
// Fix: use a smaller array which won't lead to stack-overflow. Change N to say 104.
// Alternatively declare N*N sized arrays using malloc within the 'omp parallel' construct. Malloc is a thread safe operation.
// The code below uses malloc approach. Some barriers are added just for clean output.

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;

/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid)
  {

  /* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  #pragma omp barrier
  printf("Thread %d starting...\n",tid);
  #pragma omp barrier

  /* Each thread works on its own private copy of the array */
  double *a = (double*)malloc(N*N*sizeof(double));
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i * N + j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N*N-1]);

  }  /* All threads join master thread and disband */

}

