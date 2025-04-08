//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2025) National Technology & Engineering
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

#include <Kokkos_Core.hpp>
#include <mpipcl.h>

#include <KokkosComm/traits.hpp>
#include "req.hpp"

#include "impl/types.hpp"

namespace KokkosComm {

template <typename CommSpace = DefaultCommunicationSpace>
class Channel {
 public:
  explicit Channel(int dest_rank, int src_rank, int tag, MPI_Comm comm)
      : dest_rank_(dest_rank), src_rank_(src_rank), tag_(tag), comm_(comm) {}

  // Send initialization - dynamically adds a send request to the send queue
  template <class SendView>
  void sendinit(SendView view) {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::sendinit");
    using value_type = typename SendView::value_type;
    // Add a new request to the send_reqs_ vector
    send_reqs_.emplace_back();
    MPI_Send_init(KokkosComm::data_handle(view), KokkosComm::span(view), KokkosComm::Impl::mpi_type_v<value_type>,
                  dest_rank_, tag_, comm_, &(send_reqs_.back().mpi_request()));
    Kokkos::Tools::popRegion();
  }

  // Partitioned send initialization
  template <class SendView>
  void psendinit(SendView view) {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::psendinit");
    using value_type = typename SendView::value_type;
    send_reqs_.emplace_back();
    MPIX_Psend_init(KokkosComm::data_handle(view), 1, KokkosComm::span(view), KokkosComm::Impl::mpi_type_v<value_type>,
                   dest_rank_, tag_, comm_, MPI_INFO_NULL, &(send_reqs_.back().mpix_request()));
    Kokkos::Tools::popRegion();
  }

  // Receive initialization - dynamically adds a receive request to the recv queue
  template <class RecvView>
  void recvinit(RecvView view) {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::recvinit");
    using value_type = typename RecvView::value_type;
    recv_reqs_.emplace_back();
    MPI_Recv_init(KokkosComm::data_handle(view), KokkosComm::span(view), KokkosComm::Impl::mpi_type_v<value_type>,
                  src_rank_, tag_, comm_, &(recv_reqs_.back().mpi_request()));
    Kokkos::Tools::popRegion();
  }

  // Partitioned receive initialization
  template <class RecvView>
  void precvinit(RecvView view) {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::precvinit");
    using value_type = typename RecvView::value_type;
    recv_reqs_.emplace_back();
    MPIX_Precv_init(KokkosComm::data_handle(view), 1, KokkosComm::span(view), KokkosComm::Impl::mpi_type_v<value_type>,
                  src_rank_, tag_, comm_, MPI_INFO_NULL, &(recv_reqs_.back().mpix_request()));
    Kokkos::Tools::popRegion();
  }

  void start() {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::start");
    std::vector<MPI_Request> mpi_reqs;
    for (auto& req : send_reqs_) {
      mpi_reqs.push_back(req.mpi_request());
    }
    for (auto& req : recv_reqs_) {
      mpi_reqs.push_back(req.mpi_request());
    }
    MPI_Startall(mpi_reqs.size(), mpi_reqs.data());
    Kokkos::Tools::popRegion();
  }

  void pstart() {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::pstart");
    // std::vector<MPIX_Request> mpix_reqs; //TODO: consider using Startall, similar to start()
    for (auto& req : send_reqs_) {
      MPIX_Start(&req.mpix_request());
      MPIX_Pready(0, &req.mpix_request());
    }
    MPI_Barrier(comm_);
    for (auto& req : recv_reqs_) {
      MPIX_Start(&req.mpix_request());
    }
    Kokkos::Tools::popRegion();
  }

  void wait() {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::wait");
    std::vector<Req<Mpi>> reqs;
    reqs.reserve(send_reqs_.size() + recv_reqs_.size());
    reqs.insert(reqs.end(), send_reqs_.begin(), send_reqs_.end());
    reqs.insert(reqs.end(), recv_reqs_.begin(), recv_reqs_.end());
    wait_all(reqs);
    Kokkos::Tools::popRegion();
  }

  void pwait() {
    Kokkos::Tools::pushRegion("KokkosComm::Channel::pwait");
    std::vector<MPIX_Request> mpix_reqs;
    for (auto& req : send_reqs_) {
      mpix_reqs.push_back(req.mpix_request());
    }
    for (auto& req : recv_reqs_) {
      mpix_reqs.push_back(req.mpix_request());
    }
    MPIX_Request* a = &mpix_reqs[0];
    MPIX_Waitall(mpix_reqs.size(), a, MPI_STATUSES_IGNORE);
    Kokkos::Tools::popRegion();
  }

 private:
  std::vector<Req<Mpi>> send_reqs_;  // Queue for send requests
  std::vector<Req<Mpi>> recv_reqs_;  // Queue for receive requests
  int dest_rank_;                    // Destination rank for send
  int src_rank_;                     // Source rank for receive
  int tag_;                          // MPI tag
  MPI_Comm comm_;                    // MPI communicator
};

}  // namespace KokkosComm
