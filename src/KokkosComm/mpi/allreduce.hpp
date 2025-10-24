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

#pragma once

#include <mpi.h>
#include <Kokkos_Core.hpp>

#include <KokkosComm/concepts.hpp>
#include <KokkosComm/traits.hpp>

#include "impl/types.hpp"

namespace KokkosComm::mpi {

template <KokkosView SendView, KokkosView RecvView>
void allreduce(SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allreduce");

  using SendScalar = typename SendView::value_type;
  using RecvScalar = typename RecvView::value_type;
  static_assert(std::is_same_v<std::remove_cv_t<SendScalar>, std::remove_cv_t<RecvScalar> >,
                "Send and receive views have different value types");

  //static_assert(KokkosComm::rank<SendView>() <= 1, "allreduce for SendView::rank > 1 not supported");
  //static_assert(KokkosComm::rank<RecvView>() <= 1, "allreduce for RecvView::rank > 1 not supported");

  if (!KokkosComm::is_contiguous(sv)) {
    throw std::runtime_error{"low-level allreduce requires contiguous send view"};
  }
  if (!KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error{"low-level allreduce requires contiguous recv view"};
  }
  if (sv.size() != rv.size()) {
    throw std::runtime_error{"allreduce requires send and receive views to have the same size"};
  }
  int const count = sv.size();
  MPI_Allreduce(KokkosComm::data_handle(sv), KokkosComm::data_handle(rv), count,
                KokkosComm::Impl::mpi_type_v<SendScalar>, op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosView View>
void allreduce(View const &v, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allreduce");

  using Scalar = typename View::value_type;

  //static_assert(KokkosComm::rank<View>() <= 1, "allreduce for View::rank > 1 not supported");

  if (!KokkosComm::is_contiguous(v)) {
    throw std::runtime_error("low-level allgather requires contiguous recv view");
  }
  int const count = v.size();
  MPI_Allreduce(MPI_IN_PLACE, KokkosComm::data_handle(v), count, KokkosComm::Impl::mpi_type_v<Scalar>, op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void allreduce(ExecSpace const &space, SendView const &sv, RecvView const &rv, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allreduce");

  if (!KokkosComm::is_contiguous(sv) || !KokkosComm::is_contiguous(rv)) {
    throw std::runtime_error("allreduce for non-contiguous views not implemented");
  }
  space.fence("fence before allreduce");  // work in space may have been used to produce send view data
  allreduce(sv, rv, op, comm);

  Kokkos::Tools::popRegion();
}

template <KokkosExecutionSpace ExecSpace, KokkosView View>
void allreduce(ExecSpace const &space, View const &v, MPI_Op op, MPI_Comm comm) {
  Kokkos::Tools::pushRegion("KokkosComm::mpi::allreduce");

  if (!KokkosComm::is_contiguous(v)) {
    throw std::runtime_error("allreduce for non-contiguous views not implemented");
  }
  space.fence("fence before allreduce");  // work in space may have been used to produce send view data
  allreduce(v, op, comm);

  Kokkos::Tools::popRegion();
}
}  // namespace KokkosComm::mpi
