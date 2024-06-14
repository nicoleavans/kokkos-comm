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

template<typename T>
MPI_Datatype getMPIDatatype() {
	if (std::is_same<T, float>::value){
	    return MPI_FLOAT;
	} else if (std::is_same<T, double>::value){
        return MPI_DOUBLE;
	} else if (std::is_same<T, Kokkos::complex<float>>::value) {
		return MPI_C_COMPLEX;
	} else if (std::is_same<T, Kokkos::complex<double>>::value) {
		return MPI_C_DOUBLE_COMPLEX;
	} else if (std::is_same<T, int>::value) {
        return MPI_INT;
    } else if (std::is_same<T, unsigned>::value) {
        return MPI_UNSIGNED;
    } else if (std::is_same<T, int64_t>::value) {
        return MPI_INT64_T;
    } else if (std::is_same<T, size_t>::value) {
        return MPI_UNSIGNED_LONG;
    } else {
        return MPI_DATATYPE_NULL;
    }
}

template <typename Space, typename View>
void osu_latency_Kokkos_Comm(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v){
    double t_total = 0.0;
    if(rank == 0){
        Kokkos::Timer timer;
        KokkosComm::send(space, v, 1, 0, comm);
        KokkosComm::recv(space, v, 1, 0, comm);
        t_total = timer.seconds();
        timer.reset();
    } else if (rank == 1){
        KokkosComm::recv(space, v, 0, 0, comm);
        KokkosComm::send(space, v, 0, 0, comm);
    }
    if(rank == 0){
        double latency = t_total * 1e6;
    }
}

template <typename View>
void osu_latency_MPI(benchmark::State &, MPI_Comm comm, int rank, const View &v){
    double t_start = 0.0, t_end = 0.0, t_total = 0.0;
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank == 0){
        t_start = MPI_Wtime();
        MPI_Send(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 1, 0, comm); 
        MPI_Recv(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 1, 0, comm, MPI_STATUS_IGNORE); 
        t_end = MPI_Wtime();
        t_total += (t_end - t_start);
    } else if (rank == 1){
        MPI_Recv(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 0, 0, comm, MPI_STATUS_IGNORE);
        MPI_Send(v.data(), v.size(), KokkosComm::Impl::mpi_type<typename View::value_type>(), 0, 0, comm);
    }
    if(rank == 0){
        double latency = t_total * 1e6;
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

BENCHMARK(benchmark_osu_latency_KokkosComm)->UseManualTime()->Unit(benchmark::kMicrosecond)->Arg(1)->Arg(64)->Arg(512)->Arg(4<<10)->Arg(8<<10);
BENCHMARK(benchmark_osu_latency_MPI)->UseManualTime()->Unit(benchmark::kMicrosecond)->Arg(8)->Arg(64)->Arg(512)->Arg(4<<10)->Arg(8<<10);