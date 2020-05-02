/* MPI-parallel Jacobi smoothing to solve -u''=f without blocking send and receive
 * Global vector has (Nl * 2^j, Nl * 2^j) unknowns, each processor works with its
 * part, which has (Nl, Nl) unknowns.
 */
#include <stdio.h>
#include <algorithm>
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
    MPI_Request request_out_left, request_in_left;
    MPI_Request request_out_right, request_in_right;
    MPI_Request request_out_top, request_in_top;
    MPI_Request request_out_bot, request_in_bot;

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
    double * luleft_temp = (double *) calloc(sizeof(double), Nl);
    double * luright = (double *) calloc(sizeof(double), Nl);
    double * luright_temp = (double *) calloc(sizeof(double), Nl);
    double * lutemp;

    double h = 1.0 / (Nl * p_max + 1);
    double hsq = h * h;
    double invhsq = 1./hsq;
    double gres, gres0, tol = 1e-5;

    /* initial residual */
    gres0 = compute_residual(lu, Nl, invhsq);
    if(mpirank == 0)
        printf("Iter 0: Residual: %g\n", gres0);
    gres = gres0;

    for (int iter = 0; iter < max_iters && gres/gres0 > tol; iter++) {

        /* execute jacobi on boundary points */
        for(int j = 1; j <= Nl; j++) {
            lunew[index(1,j,Nl)] = (hsq + lu[index(0,j,Nl)] + lu[index(2,j,Nl)] + lu[index(1,j-1,Nl)] + lu[index(1,j+1,Nl)])/4.0;
            lunew[index(Nl,j,Nl)] = (hsq + lu[index(Nl-1,j,Nl)] + lu[index(Nl+1,j,Nl)] + lu[index(Nl,j-1,Nl)] + lu[index(Nl,j+1,Nl)])/4.0;
        }
        for(int i = 2; i < Nl; i++) {
            lunew[index(i,1,Nl)] = (hsq + lu[index(i-1,1,Nl)] + lu[index(i+1,1,Nl)] + lu[index(i,0,Nl)] + lu[index(i,2,Nl)])/4.0;
            lunew[index(i,Nl,Nl)] = (hsq + lu[index(i-1,Nl,Nl)] + lu[index(i+1,Nl,Nl)] + lu[index(i,Nl-1,Nl)] + lu[index(i,Nl+1,Nl)])/4.0;
        }

        /* communicate the ghost values */
        /* communicate along x */
        for(int i = 1; i <= Nl; i++) {
            luleft[i-1] = lunew[index(i,1,Nl)];
            luright[i-1] = lunew[index(i,Nl,Nl)];
        }
        if(p > 1) {
            MPI_Isend(rank_j%2==0?luright:luleft, Nl, MPI_DOUBLE, rank_j%2==0?mpirank+1:mpirank-1, 123, MPI_COMM_WORLD, rank_j%2==0?&request_out_right:&request_out_left);
            MPI_Irecv(rank_j%2==0?luright_temp:luleft_temp, Nl, MPI_DOUBLE, rank_j%2==0?mpirank+1:mpirank-1, 123, MPI_COMM_WORLD, rank_j%2==0?&request_in_right:&request_in_left);
        }
        if(rank_j > 0 && rank_j < p_max-1) {
            MPI_Isend(rank_j%2==0?luleft:luright, Nl, MPI_DOUBLE, rank_j%2==0?mpirank-1:mpirank+1, 124, MPI_COMM_WORLD, rank_j%2==0?&request_out_left:&request_out_right);
            MPI_Irecv(rank_j%2==0?luleft_temp:luright_temp, Nl, MPI_DOUBLE, rank_j%2==0?mpirank-1:mpirank+1, 124, MPI_COMM_WORLD, rank_j%2==0?&request_in_left:&request_in_right);
        }
        /* communicate along y */
        if(p > 1) {
            MPI_Isend(rank_i%2==0?&lunew[index(1,1,Nl)]:&lunew[index(Nl,1,Nl)], Nl, MPI_DOUBLE, rank_i%2==0?mpirank+p_max:mpirank-p_max, 125, MPI_COMM_WORLD, rank_i%2==0?&request_out_top:&request_out_bot);
            MPI_Irecv(rank_i%2==0?&lunew[index(0,1,Nl)]:&lunew[index(Nl+1,1,Nl)], Nl, MPI_DOUBLE, rank_i%2==0?mpirank+p_max:mpirank-p_max, 125, MPI_COMM_WORLD, rank_i%2==0?&request_in_top:&request_in_bot);
        }
        if(rank_i > 0 && rank_i < p_max-1) {
            MPI_Isend(rank_i%2==0?&lunew[index(Nl,1,Nl)]:&lunew[index(1,1,Nl)], Nl, MPI_DOUBLE, rank_i%2==0?mpirank-p_max:mpirank+p_max, 126, MPI_COMM_WORLD, rank_i%2==0?&request_out_bot:&request_out_top);
            MPI_Irecv(rank_i%2==0?&lunew[index(Nl+1,1,Nl)]:&lunew[index(0,1,Nl)], Nl, MPI_DOUBLE, rank_i%2==0?mpirank-p_max:mpirank+p_max, 126, MPI_COMM_WORLD, rank_i%2==0?&request_in_bot:&request_in_top);
        }


        /* Jacobi step for inner points */
        for(int i = 2; i < Nl; i++) {
            for(int j = 2; j < Nl; j++) {
                lunew[index(i,j,Nl)] = (hsq + lu[index(i-1,j,Nl)] + lu[index(i+1,j,Nl)] + lu[index(i,j-1,Nl)] + lu[index(i,j+1,Nl)])/4.0;
            }
        }


        /* Wait for MPI requests to finish */
        if(p > 1) {
            if(rank_j < p_max-1) {
                MPI_Wait(&request_out_right, MPI_STATUS_IGNORE);
                MPI_Wait(&request_in_right, MPI_STATUS_IGNORE);
            }
            if(rank_j > 0) {
                MPI_Wait(&request_out_left, MPI_STATUS_IGNORE);
                MPI_Wait(&request_in_left, MPI_STATUS_IGNORE);
            }

            if(rank_i < p_max-1) {
                MPI_Wait(&request_out_top, MPI_STATUS_IGNORE);
                MPI_Wait(&request_in_top, MPI_STATUS_IGNORE);
            }
            if(rank_i > 0) {
                MPI_Wait(&request_out_bot, MPI_STATUS_IGNORE);
                MPI_Wait(&request_in_bot, MPI_STATUS_IGNORE);
            }
        }


        /* set the values of left and right columns */
        for(int i = 1; i <= Nl; i++) {
            if(rank_j > 0)
                lunew[index(i,0,Nl)] = luleft_temp[i-1];
            if(rank_j < p_max - 1)
                lunew[index(i,Nl+1,Nl)] = luright_temp[i-1];
        }

        /* copy newu to u using pointer flipping */
        lutemp = lu; lu = lunew; lunew = lutemp;
        if (0 == (iter % 10)) {
            gres = compute_residual(lu, Nl, invhsq);
            if (0 == mpirank) {
                printf("Iter %d: Residual: %g\n", iter + 1, gres);
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
    for(int i = 1; i <= Nl; i++) {
        for(int j = 1; j <= Nl; j++) {
            maxDiff = std::max(maxDiff, std::abs(lu[index(i,j,Nl)] - lu_seq[index(i + (p_max - 1 - rank_i) * Nl, j + rank_j * Nl, N)]));
        }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    printf("Rank %d/%d. MaxError wrt sequential = %g\n", mpirank, p, maxDiff);
    
    MPI_Finalize();
    return 0;
}
