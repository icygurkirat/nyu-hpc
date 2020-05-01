/* MPI-parallel Jacobi smoothing to solve -u''=f
 * Global vector has (Nl * 2^j, Nl * 2^j) unknowns, each processor works with its
 * part, which has (Nl, Nl) unknowns.
 */
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <string.h>

#define index(i, j, N)  ((i)*(N+2)) + (j)

/* compute global residual, assuming ghost values are updated */
double compute_residual(double *lu, int Nl, double invhsq){
    double tmp, gres = 0.0, lres = 0.0;

    for (int i = 1; i <= Nl; i++) {
        for(int j = 1; j <= Nl; j++) {
            tmp = 4 * lu[index(i,j,Nl)] - lu[index(i-1,j,Nl)] - lu[index(i,j-1,Nl)] - lu[index(i+1,j,Nl)] - lu[index(i,j+1,Nl)];
            tmp = tmp * invhsq - 1;
            lres += tmp * tmp;
        }
    }
    /* use allreduce for convenience; a reduce would also be sufficient */
    MPI_Allreduce(&lres, &gres, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres);
}


int main(int argc, char * argv[]) {
    int mpirank, p, Nl, max_iters;
    MPI_Status status, status1;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &mpirank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    /* get name of host running MPI process */
    char processor_name[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(processor_name, &name_len);
    printf("Rank %d/%d running on %s.\n", mpirank, p, processor_name);

    sscanf(argv[1], "%d", &Nl);
    sscanf(argv[2], "%d", &max_iters);

    /* check numProcs = 4^j */
    if (!((p & (p - 1) == 0) && (p & 0x55555555 > 0))) {
        printf("Exiting. p must be of the form 4^j\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    /* evaluate process indices */
    int p_max = 1, pi = p, pj = 0;
    while(pi > 1) {
        pi = pi / 4;
        p_max = p_max * 2;
    }
    pi = p % p_max;
    pj = p / p_max;

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    /* Allocation of matrix, including left, right, upper and lower ghost points */
    double * lu    = (double *) calloc(sizeof(double), (Nl + 2) * (Nl + 2));
    double * lunew = (double *) calloc(sizeof(double), (Nl + 2) * (Nl + 2));
    double * lutemp;

    double h = 1.0 / (Nl * p_max + 1);
    double hsq = h * h;
    double invhsq = 1./hsq;
    double gres, gres0, tol = 1e-5;

    /* initial residual */
    gres0 = compute_residual(lu, Nl, invhsq);
    gres = gres0;

    for (int iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

        /* Jacobi step for local points */
        for(int i = 1; i <= Nl; i++) {
            for(int j = 1; j <= Nl; j++) {
                lunew[index(i,j,Nl)] = (hsq + lu[index(i-1,j,Nl)] + lu[index(i+1,j,Nl)] + lu[index(i,j-1,Nl)] + lu[index(i,j+1,Nl)])/4.0;
            }
        }

        /* communicate ghost values */
        // send along x: even->right,left. odd->left,right
        // send along y: even->top, bot. odd->bot,top
        // if (mpirank < p - 1) {
        //     /* If not the last process, send/recv bdry values to the right */
        //     MPI_Send(&(lunew[lN]), 1, MPI_DOUBLE, mpirank+1, 124, MPI_COMM_WORLD);
        //     MPI_Recv(&(lunew[lN+1]), 1, MPI_DOUBLE, mpirank+1, 123, MPI_COMM_WORLD, &status);
        // }
        // if (mpirank > 0) {
        //     /* If not the first process, send/recv bdry values to the left */
        //     MPI_Send(&(lunew[1]), 1, MPI_DOUBLE, mpirank-1, 123, MPI_COMM_WORLD);
        //     MPI_Recv(&(lunew[0]), 1, MPI_DOUBLE, mpirank-1, 124, MPI_COMM_WORLD, &status1);
        // }


        /* copy newu to u using pointer flipping */
        lutemp = lu; lu = lunew; lunew = lutemp;
        if (0 == (iter % 10)) {
            gres = compute_residual(lu, Nl, invhsq);
            if (0 == mpirank) {
                printf("Iter %d: Residual: %g\n", iter, gres);
            }
        }
    }

    /* Clean up */
    free(lu);
    free(lunew);

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if (0 == mpirank) {
        printf("Time elapsed is %f seconds.\n", elapsed);
    }
    MPI_Finalize();
    return 0;
}
