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

int main(int argc, char *argv[])
{
    int myid, numprocs, i, j;
    int size;
    MPI_Status reqstat;
    char *s_buf, *r_buf;
    double t_start = 0.0, t_end = 0.0, t_total = 0.0;
    size_t num_elements = 0;
    MPI_Datatype omb_curr_datatype = MPI_CHAR;
    size_t omb_ddt_transmit_size = 0;
    int mpi_type_itr = 0, mpi_type_size = 0, mpi_type_name_length = 0;
    char mpi_type_name_str[OMB_DATATYPE_STR_MAX_LEN];
    MPI_Datatype mpi_type_list[OMB_NUM_DATATYPES];
    MPI_Comm omb_comm = MPI_COMM_NULL;
    omb_mpi_init_data omb_init_h;

    omb_init_h = omb_mpi_init(&argc, &argv);
    omb_comm = omb_init_h.omb_comm;
    if (MPI_COMM_NULL == omb_comm) {
        OMB_ERROR_EXIT("Cant create communicator");
    }
    MPI_CHECK(MPI_Comm_rank(omb_comm, &myid));
    MPI_CHECK(MPI_Comm_size(omb_comm, &numprocs));

    if (numprocs != 2) {
        if (myid == 0) {
            fprintf(stderr, "This test requires exactly two processes\n");
        }

        omb_mpi_finalize(omb_init_h);
        exit(EXIT_FAILURE);
    }

    /* Latency test */
    for (mpi_type_itr = 0; mpi_type_itr < options.omb_dtype_itr;
         mpi_type_itr++) {
        MPI_CHECK(MPI_Type_size(mpi_type_list[mpi_type_itr], &mpi_type_size));
        MPI_CHECK(MPI_Type_get_name(mpi_type_list[mpi_type_itr],
                                    mpi_type_name_str, &mpi_type_name_length));
        omb_curr_datatype = mpi_type_list[mpi_type_itr];
        if (0 == myid) {
            fprintf(stdout, "# Datatype: %s.\n", mpi_type_name_str);
        }
        print_only_header(myid);
        for (size = options.min_message_size; size <= options.max_message_size;
             size = (size ? size * 2 : 1)) {
            num_elements = size / mpi_type_size;
            if (0 == num_elements) {
                continue;
            }

            omb_ddt_transmit_size =
                omb_ddt_assign(&omb_curr_datatype, mpi_type_list[mpi_type_itr],
                               num_elements) *
                mpi_type_size;
            num_elements = omb_ddt_get_size(num_elements);
            set_buffer_pt2pt(s_buf, myid, options.accel, 'a', size);
            set_buffer_pt2pt(r_buf, myid, options.accel, 'b', size);

            MPI_CHECK(MPI_Barrier(omb_comm));
            t_total = 0.0;

            for (i = 0; i < options.iterations + options.skip; i++) {
                if (myid == 0) {
                    for (j = 0; j <= options.warmup_validation; j++) {
                        if (i >= options.skip &&
                            j == options.warmup_validation) {
                            t_start = MPI_Wtime();
                        }
                        // MPI_CHECK(MPI_Send(s_buf, num_elements,
                        //                    omb_curr_datatype, 1, 1, omb_comm));
                        MPI_CHECK(KokkosComm::send(Kokkos::DefaultExecutionSpace(), 1, omb_comm)); //TODO
                        // MPI_CHECK(MPI_Recv(r_buf, num_elements,
                        //                    omb_curr_datatype, 1, 1, omb_comm,
                        //                    &reqstat));
                        MPI_CHECK(KokkosComm::recv(Kokkos::DefaultExecutionSpace(), 1, omb_comm)); //TODO
                        if (i >= options.skip &&
                            j == options.warmup_validation) {
                            t_end = MPI_Wtime();
                            t_total += (t_end - t_start);
                        }
                    }
                } else if (myid == 1) {
                    for (j = 0; j <= options.warmup_validation; j++) {
                        // MPI_CHECK(MPI_Recv(r_buf, num_elements,
                        //                    omb_curr_datatype, 0, 1, omb_comm,
                        //                    &reqstat));
                        MPI_CHECK(KokkosComm::recv(Kokkos::DefaultExecutionSpace(), 1, omb_comm)); //TODO 
                        // MPI_CHECK(MPI_Send(s_buf, num_elements,
                        //                    omb_curr_datatype, 0, 1, omb_comm));
                        MPI_CHECK(KokkosComm::send(Kokkos::DefaultExecutionSpace(), 1, omb_comm)); //TODO
                    }
                }
            }

            if (myid == 0) {
                double latency = (t_total * 1e6) / (2.0 * options.iterations);
                fprintf(stdout, "%-*d", 10, size);
                fprintf(stdout, "%*.*f", FIELD_WIDTH, FLOAT_PRECISION, latency);
                fprintf(stdout, "\n");
            }
        }
    }
    omb_mpi_finalize(omb_init_h);

    return EXIT_SUCCESS;
}