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

#include "KokkosComm_collective.hpp"
#include "KokkosComm_version.hpp"
#include "KokkosComm_isend.hpp"
#include "KokkosComm_irecv.hpp"
#include "KokkosComm_recv.hpp"
#include "KokkosComm_send.hpp"
#include "KokkosComm_alltoall.hpp"
#include "KokkosComm_concepts.hpp"
#include "KokkosComm_comm_mode.hpp"

#include <Kokkos_Core.hpp>

namespace KokkosComm {

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
Req isend(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return Impl::isend<SendMode>(space, sv, dest, tag, comm);
}

template <KokkosView RecvView>
void irecv(RecvView &rv, int src, int tag, MPI_Comm comm, MPI_Request req) {
  return Impl::irecv(rv, src, tag, comm, req);
}

template <CommMode SendMode = CommMode::Default, KokkosExecutionSpace ExecSpace, KokkosView SendView>
void send(const ExecSpace &space, const SendView &sv, int dest, int tag, MPI_Comm comm) {
  return Impl::send<SendMode>(space, sv, dest, tag, comm);
}

template <KokkosExecutionSpace ExecSpace, KokkosView RecvView>
void recv(const ExecSpace &space, RecvView &rv, int src, int tag, MPI_Comm comm) {
  return Impl::recv(space, rv, src, tag, comm);
}

template <KokkosExecutionSpace ExecSpace, KokkosView SendView, KokkosView RecvView>
void alltoall(const ExecSpace &space, const SendView &sv, const size_t sendCount, const RecvView &rv,
              const size_t recvCount, MPI_Comm comm) {
  return Impl::alltoall(space, sv, sendCount, rv, recvCount, comm);
}

}  // namespace KokkosComm
