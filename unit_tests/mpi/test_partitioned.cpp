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

#include <gtest/gtest.h>
#include <type_traits>

#include <KokkosComm/KokkosComm.hpp>

namespace {

using namespace KokkosComm::mpi;

template <typename T>
class ChannelSendRecv : public testing::Test {
 public:
  using Scalar = T;
};

using ScalarTypes = ::testing::Types<int, int64_t, float, double, Kokkos::complex<float>, Kokkos::complex<double>>;
TYPED_TEST_SUITE(ChannelSendRecv, ScalarTypes);

template <typename Scalar>
void test_partitioned_channel() {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size < 2) {
    GTEST_SKIP() << "This test requires at least 2 MPI processes";
  }

  const int dest_rank = (rank + 1) % size;         // send to next rank
  const int src_rank  = (rank - 1 + size) % size;  // recv from prev rank
  const int tag       = 42;

  KokkosComm::Channel<> channel(dest_rank, src_rank, tag, MPI_COMM_WORLD);

  const int N = 10;

  // Create host view
  Kokkos::View<Scalar*, Kokkos::HostSpace> recv_host("recv_host", N);
  // Create device views
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> send_dev("send_dev", N);
  Kokkos::View<Scalar*, Kokkos::DefaultExecutionSpace> recv_dev("recv_dev", N);

  // GTEST_LOG_(INFO) << "Kokkos Execution Space: " << Kokkos::DefaultExecutionSpace::name();
  Kokkos::parallel_for(
      "init_send_dev", N, KOKKOS_LAMBDA(int i) { send_dev(i) = static_cast<Scalar>(rank * N + i); });
  Kokkos::fence();

  channel.psendinit(send_dev);
  channel.precvinit(recv_dev);

  channel.pstart();
  channel.pwait();

  Kokkos::deep_copy(recv_host, recv_dev);

  int errs = 0;
  for (int i = 0; i < N; i++) {
    const Scalar expected = static_cast<Scalar>(src_rank * N + i);
    if (recv_host(i) != expected) {
      errs++;
    }
  }
  EXPECT_EQ(errs, 0);
}

TYPED_TEST(ChannelSendRecv, 1D_contig_sendrecv_partitioned) { test_partitioned_channel<typename TestFixture::Scalar>(); }

}  // namespace
