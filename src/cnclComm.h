#include "mluTool.h"
#include "dev_MLU.h"
#include <mutex>
// RAII wrapper for CNCL communicator in a process
class CNCLComm {
public:
    // explicit防止隐式类型转化 
    explicit CNCLComm(cnclComm_t cnclComm)
        : cnclComm_(cnclComm), aborted_(false) {}

    CNCLComm() : CNCLComm(nullptr) {}

    ~CNCLComm() noexcept;
    /**
     * @brief 创建cnclComm
    */
    static std::shared_ptr<CNCLComm> create(int numRanks, int rank, int device,
                                const cnclCliqueId_t clique_id);

    // Must not be copyable
    CNCLComm(const CNCLComm&) = delete;
    CNCLComm& operator=(const CNCLComm&) = delete;

    // Do not support move assignment as there is no valid use case
    CNCLComm& operator=(CNCLComm&& other) = delete;

    // allow Move constructable
    CNCLComm(CNCLComm&& other);

    cnclComm_t getCnclComm();

    void cnclCommAbort();

    bool isAborted() const;

protected:
    cnclComm_t cnclComm_;  // 通信子对应的结构体
    bool aborted_;        // 通信子是否被中止
    mutable std::mutex mutex_; // 互斥锁
    int rank_;  // 通信子的rank号
    Dev_MLU dev_mlu; // 通信子对应的设备
};