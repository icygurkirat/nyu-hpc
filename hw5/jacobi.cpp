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
    MPI_Status status1, status2, status3, status4;

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
    if (!(((p & (p - 1)) == 0) && ((p & 0x55555555) > 0))) {
        printf("Exiting. p must be of the form 4^j\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    /* evaluate process indices */
    int p_max = 1, rank_i = p, rank_j = 0;
    while(rank_i > 1) {
        rank_i = rank_i / 4;
        p_max = p_max * 2;
    }
    rank_i = mpirank / p_max;
    rank_j = mpirank % p_max;

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    /* Allocation of matrix, including left, right, upper and lower ghost points */
    double * lu    = (double *) calloc(sizeof(double), (Nl + 2) * (Nl + 2));
    double * lunew = (double *) calloc(sizeof(double), (Nl + 2) * (Nl + 2));
    double * luleft = (double *) calloc(sizeof(double), Nl);
    double * luright = (double *) calloc(sizeof(double), Nl);
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
        /* communicate along x */
        for(int i = 1; i <= Nl; i++) {
            luleft[i-1] = lunew[index(i,1,Nl)];
            luright[i-1] = lunew[index(i,Nl,Nl)];
        }
        if(p > 1)
            MPI_Sendrecv_replace(rank_j%2==0?luright:luleft, Nl, MPI_DOUBLE, rank_j%2==0?mpirank+1:mpirank-1, 123, rank_j%2==0?mpirank+1:mpirank-1, 123, MPI_COMM_WORLD, &status1);
        if(rank_j > 0 && rank_j < p_max-1)
            MPI_Sendrecv_replace(rank_j%2==0?luleft:luright, Nl, MPI_DOUBLE, rank_j%2==0?mpirank-1:mpirank+1, 124, rank_j%2==0?mpirank-1:mpirank+1, 124, MPI_COMM_WORLD, &status2);

        /* communicate along y */
        if(p > 1)
            MPI_Sendrecv(rank_i%2==0?&lunew[index(1,1,Nl)]:&lunew[index(Nl,1,Nl)], Nl, MPI_DOUBLE, rank_i%2==0?mpirank+p_max:mpirank-p_max, 125,
                         rank_i%2==0?&lunew[index(0,1,Nl)]:&lunew[index(Nl+1,1,Nl)], Nl, MPI_DOUBLE, rank_i%2==0?mpirank+p_max:mpirank-p_max, 125, MPI_COMM_WORLD, &status3);
        if(rank_i > 0 && rank_i < p_max-1)
            MPI_Sendrecv(rank_i%2==0?&lunew[index(Nl,1,Nl)]:&lunew[index(1,1,Nl)], Nl, MPI_DOUBLE, rank_i%2==0?mpirank-p_max:mpirank+p_max, 126,
                         rank_i%2==0?&lunew[index(Nl+1,1,Nl)]:&lunew[index(0,1,Nl)], Nl, MPI_DOUBLE, rank_i%2==0?mpirank-p_max:mpirank+p_max, 126, MPI_COMM_WORLD, &status4);

        /* set the values of left and right columns */
        for(int i = 1; i <= Nl; i++) {
            lunew[index(i,0,Nl)] = luleft[i];
            lunew[index(i,Nl+1,Nl)] = luright[i];
        }

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

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if (0 == mpirank) {
        printf("Time elapsed is %f seconds.\n", elapsed);
    }


    /* Asserting against sequential implementation */
    int N = p_max * Nl;
    double * lu_seq    = (double *) calloc(sizeof(double), (N + 2) * (N + 2));
    double * lunew_seq = (double *) calloc(sizeof(double), (N + 2) * (N + 2));
    for(int iter = 0; iter < max_iters; iter++) {
        for(int i = 1; i <= N; i++) {
            for(int j = 1; j <= N; j++) {
                lunew_seq[index(i,j,N)] = (hsq + lu_seq[index(i-1,j,N)] + lu_seq[index(i+1,j,N)] + lu_seq[index(i,j-1,N)] + lu_seq[index(i,j+1,N)])/4.0;
            }
        }
        lutemp = lu_seq; lu_seq = lunew_seq; lunew_seq = lutemp;
    }
    double maxDiff = 0;
    for(int i = 1; i <= N; i++) {
        for(int j = 1; j <= N; j++) {
            maxDiff = std::max(maxDiff, std::abs(lu[index(i,j,Nl)] - lu_seq[index(i + (p_max - 1 - rank_i)*p_max, j + rank_j * p_max,N)]));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d/%d. MaxError = %g\n", mpirank, p, maxDiff);
    
    MPI_Finalize();
    return 0;
}
