/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

// Fixes:-
// tid must be private to each thread.
// Set barrier before printing that thread is starting. No change in final answer, but the actual work of all threads begins after barrier
// Variable i should be private to each thread as it sets the chunk size for each thread
// total=0.0 can be set only once, by the master thread
// Use reduction clause for finding total. Else the answer might be incorrect as all thread will be editing the global variable 'total'
int main (int argc, char *argv[]) 
{
int nthreads;
float total;

/*** Spawn parallel region ***/
#pragma omp parallel 
  {
  /* Obtain thread number */
  int tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    total = 0.0;
    printf("Number of threads = %d\n", nthreads);
    }

  #pragma omp barrier
  printf("Thread %d is starting...\n",tid);

  /* do some work */
  #pragma omp for schedule(dynamic,10) reduction(+:total)
  for (int i=0; i<1000000; i++) 
     total = total + i*1.0;

  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
  printf("Total = %e\n", total);
}
