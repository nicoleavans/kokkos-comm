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
    if(rank == 0){
        KokkosComm::send(space, v, 1, 0, comm);
        KokkosComm::recv(space, v, 1, 0, comm);
    } else if (rank == 1){
        KokkosComm::recv(space, v, 0, 0, comm);
        KokkosComm::send(space, v, 0, 0, comm);
    }
}

template <typename View>
void osu_latency_MPI(benchmark::State &, MPI_Comm comm, int rank, const View &v){
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        MPI_Send(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 1, 0, comm); 
        MPI_Recv(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 1, 0, comm, MPI_STATUS_IGNORE); 
    } else if (rank == 1){
        MPI_Recv(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 0, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 0, 0, comm);
    }
}

void benchmark_osu_latency_KokkosComm(benchmark::State &state){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 2){
        state.SkipWithError("benchmark_osu_latency_KokkosComm needs exactly 2 ranks");
    }

    auto space = Kokkos::DefaultExecutionSpace();
    using view_type = Kokkos::View<char *>;
    view_type a("", state.range(0));

    while(state.KeepRunning()){
        do_iteration(state, MPI_COMM_WORLD, osu_latency_Kokkos_Comm<Kokkos::DefaultExecutionSpace, view_type>, space, rank, a);
    }

    state.counters["bytes"] = a.size() * 2;
}

void benchmark_osu_latency_MPI(benchmark::State &state){
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); 
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if(size != 2){
        state.SkipWithError("benchmark_osu_latency_MPI needs exactly 2 ranks");
    }

    using view_type = Kokkos::View<char *>;
    view_type a("", state.range(0));

    while(state.KeepRunning()){
        do_iteration(state, MPI_COMM_WORLD, osu_latency_MPI<view_type>, rank, a);
    }

    state.counters["bytes"] = a.size() * 2;
}

BENCHMARK(benchmark_osu_latency_KokkosComm)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 32 * 1024 * 1024);
BENCHMARK(benchmark_osu_latency_MPI)->UseManualTime()->Unit(benchmark::kMicrosecond)->RangeMultiplier(2)->Range(1, 32 * 1024 * 1024);