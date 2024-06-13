//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER

// Adapted from the OSU Benchmarks
// Copyright (c) 2002-2024 the Network-Based Computing Laboratory
// (NBCL), The Ohio State University.

#include "test_utils.hpp"
#include "KokkosComm.hpp"

template <typename Space, typename View>
void osu_latency_Kokkos_Comm(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v){
    double t_total = 0.0;
    if(rank == 0){
        Kokkos::Timer timer; //TODO what to do w/ timer given benchmark utils?
        KokkosComm::send(space, v, 0, 0, comm);
        KokkosComm::recv(space, v, 0, 0, comm);
        t_total = timer.seconds();
        timer.reset();
    } else if (rank == 1){
        KokkosComm::recv(space, v, 0, 0, comm);
        KokkosComm::send(space, v, 0, 0, comm);
    }
    if(rank == 0){
        double latency = t_total * 1e6; //TODO where is 1e6 coming from?
    }
}

template <typename Space, typename View>
void osu_latency_MPI(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v){
    double t_start = 0.0, t_end = 0.0, t_total = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        t_start = MPI_Wtime();
        //MPI_Send(s_buf, num_elements, curr_datatype, 1, 1, comm); 
        //MPI_Recv(r_buf, num_elements, curr_datatype, 1, 1, comm, &reqstat); 
        t_end = MPI_Wtime();
        t_total += (t_end - t_start);
    } else if (rank == 1){
        //MPI_Recv(r_buf, num_elements, curr_datatype, 0, 1, comm, &reqstat);
        //MPI_Send(s_buf, num_elements, curr_datatype, 0, 1, comm);
    }
    if(rank == 0){
        double latency = t_total * 1e6; //TODO where is 1e6 coming from?
    }
}

void benchmark_osu_latency(benchmark::State &state){
    int rank, size;
    
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 2){
        state.SkipWithError("benchmark_osu_latency needs exactly 2 ranks");
    }

    using Scalar = double;
    auto space = Kokkos::DefaultExecutionSpace();
    using view_type = Kokkos::View<Scalar *>;

    view_type a("", 1000000);

    while(state.KeepRunning()){
        do_iteration(state, MPI_COMM_WORLD, osu_latency_Kokkos_Comm<Kokkos::DefaultExecutionSpace, view_type>, space, rank, a);
    }

    state.SetBytesProcessed(sizeof(Scalar) * state.iterations() * a.size() * 2);
}

BENCHMARK(benchmark_osu_latency)->UseManualTime()->Unit(benchmark::kMillisecond);