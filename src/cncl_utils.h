
#ifndef CNCL_UTILS_H
#define CNCL_UTILS_H
#include <stdio.h>
#include <stdlib.h>

#include <memory>
#include <vector>
#include <mutex>
#include "cncl.h"  // NOLINT

#define C10D_CNCL_CHECK(cmd)                                              \
  do {                                                                    \
    cnclResult_t error = cmd;                                             \
    if (error != CNCL_RET_SUCCESS) {                                      \
      std::string err = "CNCL error in: " + std::string(__FILE__) + ":" + \
                        std::to_string(__LINE__) + ", " +                 \
                        std::string(cnclGetErrorStr(error));              \
      TORCH_CHECK(false, err);                                            \
    }                                                                     \
  } while (0)

#define C10D_CNCL_ASSERT(cmd)                                           \
  do {                                                                  \
    cnclResult_t res = cmd;                                             \
    if (res != CNCL_RET_SUCCESS) {                                      \
      std::string err = cnclGetErrorStr(res);                           \
      fprintf(stderr, "CNCL error in: %s:%d, %s\n", __FILE__, __LINE__, \
              err.c_str());                                             \
      abort();                                                          \
    }                                                                   \
  } while (0)

namespace torch_mlu {

// RAII wrapper for CNCL communicator in a process
class CNCLComm {
 public:
  explicit CNCLComm(cnclComm_t cnclComm)  // NOSONAR
      : cnclComm_(cnclComm), aborted_(false) {}

  CNCLComm() : CNCLComm(nullptr) {}

  ~CNCLComm() noexcept {
    // Add lock in this destructor, as aborted_ needs to be read after memory
    // barrier here.
    std::unique_lock<std::mutex> lock(mutex_);
    if (cnclComm_ && !aborted_) {
      // TODO(zhiguangda): use cnclCommAbort when catch support
      // environment variable like ENABLE_NCCL_ERROR_CHECKING
      C10D_CNCL_ASSERT(cnclDestroyComms(&cnclComm_, 1));
    }
  }

  static std::shared_ptr<CNCLComm> create(int numRanks, int rank, int device,
                                          const cnclCliqueId_t clique_id) {
    auto comm = std::make_shared<CNCLComm>();
    C10D_CNCL_CHECK(cnclInitComms(&(comm->cnclComm_), 1, &device, &rank,
                                  numRanks, clique_id));
    comm->rank_ = rank;
    return comm;
  }

  // Must not be copyable
  CNCLComm(const CNCLComm&) = delete;
  CNCLComm& operator=(const CNCLComm&) = delete;

  // Do not support move assignment as there is no valid use case
  CNCLComm& operator=(CNCLComm&& other) = delete;

  // Move constructable
  CNCLComm(CNCLComm&& other) {  // NOSONAR
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(cnclComm_, other.cnclComm_);
    std::swap(aborted_, other.aborted_);
  }

  cnclComm_t getCnclComm() { return cnclComm_; };

  void cnclCommAbort() {
    std::unique_lock<std::mutex> lock(mutex_);
    if (aborted_) {
      // Should not abort twice.
      return;
    }

    C10D_CNCL_CHECK(cnclAbortComm(cnclComm_));
    aborted_ = true;
    cnclComm_ = nullptr;
  }

  bool isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
  }

 protected:
  cnclComm_t cnclComm_;
  bool aborted_;
  mutable std::mutex mutex_;
  // Rank that this communicator corresponds to.
  int rank_;
};

}  // namespace torch_mlu
#endif
