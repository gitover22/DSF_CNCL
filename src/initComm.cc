#include "initComm.h"

bool InitComm(cnclComm_t *comms, int num_comm,
              const int *dev_list, const int *rank_list, int nrank, cnclCliqueId_t clique_id)
{

    CNCL_CHECK(cnclInitComms(comms, num_comm, dev_list, rank_list, nrank, clique_id));
    return true;
}

bool Map_Comm(cnclComm_t *tComm_list, std::vector<CNCLComm> &comm_list, int comms)
{
    if (tComm_list == nullptr || comm_list.empty())
        return false;
    // 遍历数组并初始化每个元素
    for (int i = 0; i < comms; ++i)
    {
        comm_list[i].cnclComm_ = tComm_list[i]; // 复制结构
    }

    return true;
}

bool Vertify_CnclComm(std::vector<CNCLComm> &comm_list, int comms, std::vector<Dev_MLU> &dev_list)
{
    CNCL_CHECK(cnclSend(dev_list[2].get_send_buffer(), 256, cnclInt32, comm_list[7].get_rank(),
                        comm_list[2].getCnclComm(), dev_list[2].get_queue()));
    CNCL_CHECK(cnclRecv(dev_list[7].get_recv_buffer(), 256, cnclInt32, comm_list[2].get_rank(),
                        comm_list[7].getCnclComm(), dev_list[7].get_queue()));

    return true;
}
/**
 * @brief 交换缓冲区数据
 * @param m [in] 发送数据的mlu下标
 * @param n [in] 接受数据的mlu下标
 * @param send_buffer [in] 所有设备的发送缓冲区
 * @param recv_buffer [in] 所有设备的接收缓冲区
 */
void operator_buffer_data(int m, int n, void **send_buffer, void **recv_buffer, cnclComm_t *comms, cnrtQueue_t *queues, int COUNT)
{
    CNCL_CHECK(cnclSend(send_buffer[m], COUNT, cnclInt32, n, comms[m], queues[m]));
    CNCL_CHECK(cnclRecv(recv_buffer[n], COUNT, cnclInt32, m, comms[n], queues[n]));
}