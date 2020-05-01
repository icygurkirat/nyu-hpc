/* Illustrate exchange of array of doubles
 * ping-pong style between even and odd processors
 * every processor writes its result to a file
 */
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main( int argc, char *argv[]) {
  int rank, n;
  MPI_Status status;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  int N = 10;

  double *message_out = (double*) calloc(N, sizeof(double));
  double *message_in = (double*) calloc(N, sizeof(double));

  for(n = 0; n < N; ++n)
    message_out[n] = rank;

  int tag = 99;
  int origin, destination;

  if(rank % 2 == 0) {
    destination = rank + 1;
    origin = rank + 1;

    MPI_Send(message_out, N, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
    MPI_Recv(message_in,  N, MPI_DOUBLE, origin,      tag, MPI_COMM_WORLD, &status);
  } else {
    destination = rank - 1;
    origin = rank - 1;

    MPI_Recv(message_in,  N, MPI_DOUBLE, origin,      tag, MPI_COMM_WORLD, &status);
    MPI_Send(message_out, N, MPI_DOUBLE, destination, tag, MPI_COMM_WORLD);
  }

  { // Write output to a file
    FILE* fd = NULL;
    char filename[256];
    snprintf(filename, 256, "output%02d.txt", rank);
    fd = fopen(filename,"w+");

    if(NULL == fd) {
      printf("Error opening file \n");
      return 1;
    }

    fprintf(fd, "rank %d received from %d the message:\n", rank, origin);
    for(n = 0; n < N; ++n)
      fprintf(fd, "  %f\n", message_in[n]);

    fclose(fd);
  }

  MPI_Finalize();
  return 0;
}
