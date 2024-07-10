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

#include <memory>
#include <vector>

#include "KokkosComm_include_mpi.hpp"

namespace KokkosComm {

class Req {
  // a type-erased view. Request uses these to keep temporary views alive for
  // the lifetime of "Immediate" MPI operations
  struct ViewHolderBase {
    virtual ~ViewHolderBase() {}
  };
  template <typename V>
  struct ViewHolder : ViewHolderBase {
    ViewHolder(const V &v) : v_(v) {}
    V v_;
  };

  struct Record {
    Record() : req_(MPI_REQUEST_NULL) {}
    MPI_Request req_;
    std::vector<std::shared_ptr<ViewHolderBase>> until_waits_;
    std::vector<std::function<void()>> functions_after_wait_;
  };

 public:
  Req() : record_(std::make_shared<Record>()) {}

  MPI_Request &mpi_req() { return record_->req_; }

  void wait() {
    MPI_Wait(&(record_->req_), MPI_STATUS_IGNORE);
    for (auto &f : record_->functions_after_wait_) {
      f();
    }
    record_->until_waits_.clear();  // drop any views we're keeping alive until wait()
    record_->functions_after_wait_.clear();
  }

  // keep a reference to this view around until wait() is called
  template <typename View>
  void keep_until_wait(const View &v) {
    record_->until_waits_.push_back(std::make_shared<ViewHolder<View>>(v));
  }

  void call_after_wait(std::function<void()> &&f) {
    record_->functions_after_wait_.push_back(f);
  }

 private:
  std::shared_ptr<Record> record_;
};

}  // namespace KokkosComm
