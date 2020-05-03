// Parallel sample sort
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <algorithm>

int main( int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, p;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &p);

    // Number of random numbers per processor (this should be increased
    // for actual tests or could be passed in through the command line
    int N = 100;
    if(argc > 1)
        N = atoi(argv[1]);

    int* vec = (int*)malloc(N*sizeof(int));
    // seed random number generator differently on every core
    srand((unsigned int) (rank + 393919));

    // fill vector with random integers
    for (int i = 0; i < N; ++i)
        vec[i] = rand();

    printf("Rank: %d/%d, first entry: %d\n", rank, p, vec[0]);
    MPI_Barrier(MPI_COMM_WORLD);
    double tt = MPI_Wtime();

    // sort locally
    std::sort(vec, vec+N);

    // sample p-1 entries from vector as the local splitters, i.e.,
    // every N/P-th entry of the sorted vector
    int* sample = (int*)malloc((p-1)*sizeof(int));
    for(int i = 1; i < p; i++)
        sample[i-1] = vec[(i*N)/p];

    // every process communicates the selected entries to the root
    // process; use for instance an MPI_Gather
    int *superSample;
    if(rank == 0)
        superSample = (int*)malloc(p*(p-1)*sizeof(int));
    MPI_Gather(sample, p-1, MPI_INTEGER, superSample, p-1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    // root process does a sort and picks (p-1) splitters (from the
    // p(p-1) received elements)
    int *splitter = (int*)malloc((p-1)*sizeof(int));
    if(rank == 0) {
        std::sort(superSample, superSample+p*(p-1));
        for(int i = 1; i < p ; i++)
            splitter[i-1] = superSample[i*(p-1)];
    }

    // root process broadcasts splitters to all other processes
    MPI_Bcast(splitter, p-1, MPI_INTEGER, 0, MPI_COMM_WORLD);

    // communicate the send-counts
    int* sendCount = (int*)malloc(sizeof(int)*p);
    int* recvCount = (int*)malloc(sizeof(int)*p);
    sendCount[0] = -1;
    for(int i = 1; i < p; i++)
        sendCount[i] = std::lower_bound(vec, vec+N, splitter[i-1]) - vec;
    for(int i = 0; i < p-1; i++)
        sendCount[i] = sendCount[i+1] - sendCount[i];
    sendCount[p-1] = N - 1 - sendCount[p-1];
    MPI_Alltoall(sendCount, 1, MPI_INTEGER, recvCount, 1, MPI_INTEGER, MPI_COMM_WORLD);

    // communicate the exact data
    int N2=0;
    for(int i = 0; i < p; i++)
        N2 += recvCount[i];
    int* vec2 = (int*)malloc(N2*sizeof(int));
    int* sendDisp = (int*)malloc(sizeof(int)*p);
    int* recvDisp = (int*)malloc(sizeof(int)*p);
    sendDisp[0] = recvDisp[0] = 0;
    for(int i = 1; i < p; i++) {
        sendDisp[i] = sendCount[i-1];
        recvDisp[i] = recvCount[i-1];
    }

    MPI_Alltoallv(vec, sendCount, sendDisp, MPI_INTEGER, vec2, recvCount, recvDisp, MPI_INTEGER, MPI_COMM_WORLD);


    // do a local sort of the received data
    std::sort(vec2, vec2+N2);

    // every process writes its result to a file

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsed = MPI_Wtime() - tt;
    if (0 == rank) {
        printf("Time elapsed is %f seconds.\n", elapsed);
    }

    // Assert the correct ordering
    bool flag = false;
    for(int i = 1; i < N2; i++) {
        if(vec2[i] < vec2[i-1]){
            flag = true;
            break;
        }
    }
    if(flag)
        printf("Rank %d/%d: ASSERTION_FAILURE. PartitionSize = %d\n", rank, p, N2);
    else
        printf("Rank %d/%d: ASSERTION_SUCCESSFUL. MaxValue = %d. PartitionSize = %d\n", rank, p, vec2[N2-1], N2);


    // Write output to a file
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }

    for(int i = 0; i < N2; i++)
      fprintf(fd, "%d  ", vec2[i]);

    fclose(fd);

    free(vec);
    MPI_Finalize();
    return 0;
}
