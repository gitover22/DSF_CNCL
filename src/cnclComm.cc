#include "cnclComm.h"

CNCLComm::~CNCLComm() noexcept {
    // Add lock in this destructor, as aborted_ needs to be read after memory
    // barrier here.
    std::unique_lock<std::mutex> lock(mutex_);
    if (cnclComm_ && !aborted_) {
        // TODO(zhiguangda): use cnclCommAbort when catch support
        // environment variable like ENABLE_NCCL_ERROR_CHECKING
        // CNCL_ASSERT(cnclDestroyComms(&cnclComm_, 1));
    }
    
}

void CNCLComm::init_comm(int total_Ranks, int device,int rank, const cnclCliqueId_t clique_id) {
    printf("=== debug2 ===\n");
    // CNCL_CHECK(cnclInitComms(&cnclComm_, 1, &device, &rank, total_Ranks, clique_id));
    CNCL_CHECK(cnclInitComms(&cnclComm_, 1, &device, &rank, 1, clique_id));
    printf("=== debug2 ===\n");
}

// Move constructable
CNCLComm::CNCLComm(CNCLComm&& other) {  // NOSONAR
    // Using other's lock, as it reads other's states
    // Can not use this.mutex_, as this object is being constructed.
    std::unique_lock<std::mutex> lock(other.mutex_);
    std::swap(cnclComm_, other.cnclComm_);
    std::swap(aborted_, other.aborted_);
    // std::swap(dev_mlu, other.dev_mlu);
}

cnclComm_t CNCLComm::getCnclComm() { return cnclComm_; };

void CNCLComm::cnclCommAbort() {
    std::unique_lock<std::mutex> lock(mutex_);  
    if (aborted_) {
        // Should not abort twice.
        return;
    }
    CNCL_CHECK(cnclAbortComm(cnclComm_));
    aborted_ = true;
    cnclComm_ = nullptr;
}

bool CNCLComm::isAborted() const {
    std::unique_lock<std::mutex> lock(mutex_);
    return aborted_;
}

void CNCLComm::init_rank(int rank){
    this->rank_ = rank;
}

int CNCLComm::get_rank(){
    return this->rank_;
}
