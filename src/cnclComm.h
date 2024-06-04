#ifndef CNCL_COMM_H
#define CNCL_COMM_H
#include "mluTool.h"
#include "dev_MLU.h"
#include <mutex>


class CNCLComm;
bool Map_Comm(cnclComm_t *tComm_list, std::vector<CNCLComm> &comm_list, int comms);
// RAII wrapper for CNCL communicator in a process
class CNCLComm {
public:
    // explicit防止隐式类型转化 
    explicit CNCLComm(cnclComm_t cnclComm)
        : cnclComm_(cnclComm), aborted_(false) {}
    CNCLComm(int rank_id) : rank_(rank_id) {}
    CNCLComm() : CNCLComm(nullptr) {}

    ~CNCLComm() noexcept;
    /**
     * @brief 创建cnclComm
    */
    void init_comm(int total_Ranks, int device,int rank, const cnclCliqueId_t clique_id);

    // Must not be copyable
    CNCLComm(const CNCLComm&) = delete;
    CNCLComm& operator=(const CNCLComm&) = delete;

    // Do not support move assignment as there is no valid use case
    CNCLComm& operator=(CNCLComm&& other) = delete;

    // allow Move constructable
    CNCLComm(CNCLComm&& other);

    void init_rank(int rank);

    int get_rank();

    cnclComm_t getCnclComm();

    void cnclCommAbort();

    bool isAborted() const;
    friend bool Map_Comm(cnclComm_t *tComm_list, std::vector<CNCLComm> &comm_list, int comms);
protected:
    cnclComm_t cnclComm_;  // 通信子对应的结构体
    bool aborted_;        // 通信子是否被中止
    mutable std::mutex mutex_; // 互斥锁
    int rank_;  // 通信子的rank号
    Dev_MLU dev_mlu; // 通信子对应的设备
};


#endif