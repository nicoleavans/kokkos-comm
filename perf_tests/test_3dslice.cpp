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

#include "test_utils.hpp"
#include "KokkosComm.hpp"

template <typename Space, typename View>
void send_recv_slice_deepcopy_contig(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v) {
  // Create a slice of the 3D view
  auto sub_view = Kokkos::subview(v, 0, 0, Kokkos::ALL); // contig

  if (0 == rank) {
    KokkosComm::Req sendreq = KokkosComm::isend<KokkosComm::Impl::Packer::DeepCopy<decltype(sub_view)>>(space, sub_view, 1, 0, comm);
    sendreq.wait();
  } 
  else if (1 == rank) {
    KokkosComm::Req recvreq = KokkosComm::irecv<KokkosComm::Impl::Packer::DeepCopy<decltype(sub_view)>>(space, sub_view, 0, 0, comm, NULL);
    recvreq.wait();
  }
}

template <typename Space, typename View>
void send_recv_slice_deepcopy_noncontig(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v) {
  // Create a slice of the 3D view
  auto sub_view = Kokkos::subview(v, 0, Kokkos::ALL, 0); // noncontig

  if (0 == rank) {
    KokkosComm::Req sendreq = KokkosComm::isend<KokkosComm::Impl::Packer::DeepCopy<decltype(sub_view)>>(space, sub_view, 1, 0, comm);
    sendreq.wait();
  } 
  else if (1 == rank) {
    KokkosComm::Req recvreq = KokkosComm::irecv<KokkosComm::Impl::Packer::DeepCopy<decltype(sub_view)>>(space, sub_view, 0, 0, comm, NULL);
    recvreq.wait();
  }
}

template <typename Space, typename View>
void send_recv_slice_deepcopy_noncontig2D(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v) {
  // Create a slice of the 3D view
  auto sub_view = Kokkos::subview(v, 0, Kokkos::ALL, Kokkos::ALL); //Non-contig. memory

  if (0 == rank) {
    KokkosComm::Req sendreq = KokkosComm::isend<KokkosComm::Impl::Packer::DeepCopy<decltype(sub_view)>>(space, sub_view, 1, 0, comm);
    sendreq.wait();
  } 
  else if (1 == rank) {
    KokkosComm::Req recvreq = KokkosComm::irecv<KokkosComm::Impl::Packer::DeepCopy<decltype(sub_view)>>(space, sub_view, 0, 0, comm, NULL);
    recvreq.wait();
  }
}

template <typename Space, typename View>
void send_recv_slice_datatype_contig(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v) {
  // Create a slice of the 3D view
  auto sub_view = Kokkos::subview(v, 0, 0, Kokkos::ALL); // contig

  if (0 == rank) {
    KokkosComm::Req sendreq = KokkosComm::isend<KokkosComm::Impl::Packer::MpiDatatype<decltype(sub_view)>>(space, sub_view, 1, 0, comm);
    sendreq.wait();
  } 
  else if (1 == rank) {
    KokkosComm::Req recvreq = KokkosComm::irecv<KokkosComm::Impl::Packer::MpiDatatype<decltype(sub_view)>>(space, sub_view, 0, 0, comm, NULL);
    recvreq.wait();
  }
}

template <typename Space, typename View>
void send_recv_slice_datatype_noncontig(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v) {
  // Create a slice of the 3D view
  auto sub_view = Kokkos::subview(v, 0, Kokkos::ALL, 0); // noncontig

  if (0 == rank) {
    KokkosComm::Req sendreq = KokkosComm::isend<KokkosComm::Impl::Packer::MpiDatatype<decltype(sub_view)>>(space, sub_view, 1, 0, comm);
    sendreq.wait();
  } 
  else if (1 == rank) {
    KokkosComm::Req recvreq = KokkosComm::irecv<KokkosComm::Impl::Packer::MpiDatatype<decltype(sub_view)>>(space, sub_view, 0, 0, comm, NULL);
    recvreq.wait();
  }
}

template <typename Space, typename View>
void send_recv_slice_datatype_noncontig2D(benchmark::State &, MPI_Comm comm, const Space &space, int rank, const View &v) {
  // Create a slice of the 3D view
  auto sub_view = Kokkos::subview(v, 0, Kokkos::ALL, Kokkos::ALL); //Non-contig. memory

  if (0 == rank) {
    KokkosComm::Req sendreq = KokkosComm::isend<KokkosComm::Impl::Packer::MpiDatatype<decltype(sub_view)>>(space, sub_view, 1, 0, comm);
    sendreq.wait();
  } 
  else if (1 == rank) {
    KokkosComm::Req recvreq = KokkosComm::irecv<KokkosComm::Impl::Packer::MpiDatatype<decltype(sub_view)>>(space, sub_view, 0, 0, comm, NULL);
    recvreq.wait();
  }
}

void benchmark_3dslice_deepcopy_contig(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_3dview needs at least 2 ranks");
  }

  auto space = Kokkos::DefaultExecutionSpace();
  Kokkos::View<double ***> a("3DView", state.range(0), state.range(0), state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, send_recv_slice_deepcopy_contig<Kokkos::DefaultExecutionSpace, Kokkos::View<double ***>>, space,
                 rank, a);
  }
  state.SetBytesProcessed(sizeof(double) * state.iterations() * a.size());
  state.counters["bytes"] = state.range(0);
}

void benchmark_3dslice_deepcopy_noncontig(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_3dview needs at least 2 ranks");
  }

  auto space = Kokkos::DefaultExecutionSpace();
  Kokkos::View<double ***> a("3DView", state.range(0), state.range(0), state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, send_recv_slice_deepcopy_noncontig<Kokkos::DefaultExecutionSpace, Kokkos::View<double ***>>, space,
                 rank, a);
  }
  state.SetBytesProcessed(sizeof(double) * state.iterations() * a.size());
  state.counters["bytes"] = state.range(0);
}

void benchmark_3dslice_deepcopy_noncontig2D(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_3dview needs at least 2 ranks");
  }

  auto space = Kokkos::DefaultExecutionSpace();
  Kokkos::View<double ***> a("3DView", state.range(0), state.range(0), state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, send_recv_slice_deepcopy_noncontig2D<Kokkos::DefaultExecutionSpace, Kokkos::View<double ***>>, space,
                 rank, a);
  }
  state.SetBytesProcessed(sizeof(double) * state.iterations() * a.size());
  state.counters["bytes"] = state.range(0);
}

void benchmark_3dslice_datatype_contig(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_3dview needs at least 2 ranks");
  }

  auto space = Kokkos::DefaultExecutionSpace();
  Kokkos::View<double ***> a("3DView", state.range(0), state.range(0), state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, send_recv_slice_datatype_contig<Kokkos::DefaultExecutionSpace, Kokkos::View<double ***>>, space,
                 rank, a);
  }
  state.SetBytesProcessed(sizeof(double) * state.iterations() * a.size());
  state.counters["bytes"] = state.range(0);
}

void benchmark_3dslice_datatype_noncontig(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_3dview needs at least 2 ranks");
  }

  auto space = Kokkos::DefaultExecutionSpace();
  Kokkos::View<double ***> a("3DView", state.range(0), state.range(0), state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, send_recv_slice_datatype_noncontig<Kokkos::DefaultExecutionSpace, Kokkos::View<double ***>>, space,
                 rank, a);
  }
  state.SetBytesProcessed(sizeof(double) * state.iterations() * a.size());
  state.counters["bytes"] = state.range(0);
}

void benchmark_3dslice_datatype_noncontig2D(benchmark::State &state) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  if (size < 2) {
    state.SkipWithError("benchmark_3dview needs at least 2 ranks");
  }

  auto space = Kokkos::DefaultExecutionSpace();
  Kokkos::View<double ***> a("3DView", state.range(0), state.range(0), state.range(0));

  while (state.KeepRunning()) {
    do_iteration(state, MPI_COMM_WORLD, send_recv_slice_datatype_noncontig2D<Kokkos::DefaultExecutionSpace, Kokkos::View<double ***>>, space,
                 rank, a);
  }
  state.SetBytesProcessed(sizeof(double) * state.iterations() * a.size());
  state.counters["bytes"] = state.range(0);
}

BENCHMARK(benchmark_3dslice_deepcopy_contig)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 1024);

BENCHMARK(benchmark_3dslice_deepcopy_noncontig)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 1024);

BENCHMARK(benchmark_3dslice_deepcopy_noncontig2D)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 1024);

BENCHMARK(benchmark_3dslice_datatype_contig)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 1024);

BENCHMARK(benchmark_3dslice_datatype_noncontig)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 1024);

BENCHMARK(benchmark_3dslice_datatype_noncontig2D)
    ->UseManualTime()
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(4)
    ->Range(1, 1024);