#ifndef INITCOMM_H
#define INITCOMM_H

#include "cncl.h"
#include "cnrt.h"
#include "mluTool.h"
#include "cnclComm.h"
/**
 * @brief Initialize the CNCL communicator
 * @return true if the CNCL communicator is initialized successfully, false otherwise
*/
bool InitComm(cnclComm_t *comms, int num_comm, \
    const int *dev_list, const int *rank_list, int nrank, cnclCliqueId_t clique_id);


/**
 * @brief Map the CNCL communicator to CNCLComm object
*/
bool Map_Comm(cnclComm_t *tComm_list, std::vector<CNCLComm> &comm_list, int comms);

/**
 * @brief 验证cnclComm_t是否成功初始化
 * @param comm_list [in] 待验证的cnclComm_t数组
 * @param comms [in] 待验证的cnclComm_t数组长度
*/
bool Vertify_CnclComm(std::vector<CNCLComm> &comm_list,int comms,std::vector<Dev_MLU> &dev_list);

/**
 * @brief 交换缓冲区数据
 * @param m [in] 发送数据的mlu下标
 * @param n [in] 接受数据的mlu下标
 * @param send_buffer [in] 所有设备的发送缓冲区
 * @param recv_buffer [in] 所有设备的接收缓冲区
*/
void operator_buffer_data(int m,int n,void **send_buffer,void **recv_buffer,cnclComm_t *comms,cnrtQueue_t *queues,int COUNT);
#endif