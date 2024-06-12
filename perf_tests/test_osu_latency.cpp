/*
 * Copyright (c) 2002-2024 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include "KokkosComm.hpp"
#include <iostream> //necessary?
#include "test_utils.hpp" //MAYBE

void benchmark_latency(benchmark::State &state){

}

int main(int argc, char *argv[])
{
    int rank, numprocs, i, j;
    int size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0, t_total = 0.0;
    size_t num_elements = 0;
    MPI_Datatype curr_datatype = MPI_CHAR;
    int mpi_type_itr = 0, mpi_type_size = 0;
    MPI_Comm comm = MPI_COMM_NULL; //TODO 

    if (MPI_COMM_NULL == comm) {
        //state.SkipWithError("Cant create communicator"); //TODO
    }
    MPI_Comm_rank(comm, &rank); 
    MPI_Comm_size(comm, &numprocs);

    if (numprocs != 2) {
        if (rank == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }
        exit(EXIT_FAILURE);
    }

    /* Latency test */
    for (mpi_type_itr = 0; mpi_type_itr < options.omb_dtype_itr; mpi_type_itr++) { //TODO
        for (size = options.min_message_size; size <= options.max_message_size; //TODO
             size = (size ? size * 2 : 1)) {
            num_elements = size / mpi_type_size;
            if (0 == num_elements) {
                continue;
            }

            MPI_Barrier(comm);
            t_total = 0.0;

            for (i = 0; i < options.iterations + options.skip; i++) {
                if (rank == 0) {
                    for (j = 0; j <= options.warmup_validation; j++) {
                        if (i >= options.skip &&
                            j == options.warmup_validation) {
                            t_start = MPI_Wtime();
                        }
                        // MPI_Send(s_buf, num_elements, curr_datatype, 1, 1, comm); 
                        KokkosComm::send(Kokkos::DefaultExecutionSpace(), 1, comm);
                        // MPI_Recv(r_buf, num_elements, curr_datatype, 1, 1, comm, &reqstat); 
                        KokkosComm::recv(Kokkos::DefaultExecutionSpace(), 1, comm);
                        if (i >= options.skip && j == options.warmup_validation) {
                            t_end = MPI_Wtime();
                            t_total += (t_end - t_start);
                        }
                    }
                } else if (rank == 1) {
                    for (j = 0; j <= options.warmup_validation; j++) {
                        // MPI_Recv(r_buf, num_elements, curr_datatype, 0, 1, comm, &reqstat); 
                        KokkosComm::recv(Kokkos::DefaultExecutionSpace(), 1, comm);
                        // MPI_Send(s_buf, num_elements, curr_datatype, 0, 1, comm); 
                        KokkosComm::send(Kokkos::DefaultExecutionSpace(), 1, comm);
                    }
                }
            }

            if (rank == 0) {
                double latency = (t_total * 1e6) / (2.0 * options.iterations); //TODO replace options.iterations
                std::cout << size << ' ' << latency << '\n';
            }
        }
    }
}