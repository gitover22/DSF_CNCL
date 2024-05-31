#include "initComm.h"

bool InitComm(cnclComm_t *comms, int num_comm, \
    const int *dev_list, const int *rank_list, int nrank, cnclCliqueId_t clique_id){

    CNCL_CHECK(cnclInitComms(comms, num_comm,dev_list, rank_list, nrank,clique_id));
    return true;
}


bool Map_Comm(cnclComm_t *tComm_list,CNCLComm* comm_list,int comms){
    if (tComm_list == nullptr || comm_list == nullptr) return false;
    // 遍历数组并初始化每个元素
    for (int i = 0; i < comms; ++i) {
        comm_list[i].cnclComm_ = tComm_list[i];  // 复制结构
    }

    return true;
}
