/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

// sum must not be declared within the dotprod() because it means that it should be private. But reduction variables are required to be shared.
// sum can also not be declared within main, because it will not be in the scope of dotprod() function
// Fix: delcare sum as a global variable. Set its value to zero in main()

float a[VECLEN], b[VECLEN];
float sum;

float dotprod ()
{
int i,tid;

tid = omp_get_thread_num();
#pragma omp for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
}


int main (int argc, char *argv[]) {
int i;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
sum = 0.0;

#pragma omp parallel shared(sum)
  dotprod();

printf("Sum = %f\n",sum);

}

