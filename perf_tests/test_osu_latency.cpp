/*
 * Copyright (c) 2002-2024 the Network-Based Computing Laboratory
 * (NBCL), The Ohio State University.
 *
 * Contact: Dr. D. K. Panda (panda@cse.ohio-state.edu)
 *
 * For detailed copyright and licensing information, please refer to the
 * copyright file COPYRIGHT in the top level OMB directory.
 */
#include <iostream> //necessary?

#include "KokkosComm.hpp"
#include "test_utils.hpp" // standardize tests with others, rather than OSU utils

void benchmark_osu_latency(benchmark::State &state){
    int rank, size;
    double t_start = 0.0, t_end = 0.0, t_total = 0.0;
    //TODO consider where these come from, command line, constant, etc.
    int msg_size, min_msg_size, max_msg_size;
    int epochs, iter;

    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size < 2){
        //previously required exactly two, verify
        state.SkipWithError("benchmark_osu_latency needs at least 2 ranks");
    }
    
    for(int i = 0; i < epochs; i++){
        for(msg_size = min_msg_size; msg_size <= max_msg_size; 
            msg_size = (msg_size ? msg_size * 2 : 1)){
                MPI_Barrier(MPI_COMM_WORLD);
                t_total = 0.0;

                for(i = 0; i < iter; i++){ //TODO replace options.skip?
                    if(rank == 0){
                        
                    }
                }
            }
    }
}

int main(int argc, char *argv[]){
    int rank, size, i, j;
    int msg_size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0, t_total = 0.0;
    size_t num_elements = 0;
    MPI_Datatype curr_datatype = MPI_CHAR;
    mpi_type_size = 0;
    MPI_Comm comm = MPI_COMM_NULL; //TODO

    if (MPI_COMM_NULL == comm) {
        std::cerr << "Cant create communicator" << '\n';
    }
    MPI_Comm_rank(comm, &rank); 
    MPI_Comm_size(comm, &size);

    if (size != 2) {
        if (rank == 0) {
            std::cerr << "This test requires exactly two processes" << '\n';
        }
    }

    /* Latency test */
    for (int i = 0; i < options.omb_dtype_itr; i++) { //TODO
        for (msg_size = options.min_message_size; msg_size <= options.max_message_size; //TODO
             msg_size = (msg_size ? msg_size * 2 : 1)) {
            num_elements = msg_size / mpi_type_size;

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
                std::cout << msg_size << ' ' << latency << '\n';
            }
        }
    }
}