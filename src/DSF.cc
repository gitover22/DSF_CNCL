/**
 * @brief test for mlu comm
 * @author huafeng
 */
#include <memory>
#include <thread>
#include <iostream>
#include "mluTool.h"
#include "dev_MLU.h"
#include "cnclComm.h"
#include "initComm.h"
/**
 * 一个通信子对应一个设备，通信子用rank号来进行标记
 */
int main(int argc, char *argv[])
{
    uint32_t local_mlu_num = 0; // mlu数量
    GetMluNums(&local_mlu_num);
    printf("local mlu nums: %d\n", local_mlu_num);
    int comms; // 通信子数量
    printf("please input comm numbers (less or equal to mlu number): ");
    scanf("%d", &comms);
    getchar();
    // Dev_MLU dev_list[comms];
    std::vector<Dev_MLU> VDev;
    VDev.reserve(comms); // 预先分配足够空间

    // CNCLComm comm_list[comms];
    std::vector<CNCLComm> VComm;
    VComm.reserve(comms);
    cnclComm_t *tComm_list = new cnclComm_t[comms]; // communicator list
    int *tDev_list = new int[comms];
    int *tRank_list = new int[comms];
    MapRankandDev(comms, local_mlu_num, tDev_list, tRank_list);

    for (int i = 0; i < comms; i++)
    {
        // comm_list[i].init_rank(i);
        VComm.emplace_back(i);
        VDev.emplace_back(VComm[i].get_rank() % local_mlu_num, 1, 1, true);
    }
    cnclCliqueId clique_id;
    CNCL_CHECK(cnclGetCliqueId(&clique_id)); // 如何通过通信域id获取通信域内的通信子数量？
    // 初始化通信子
    InitComm(tComm_list, comms, tDev_list, tRank_list, comms, &clique_id);

    Map_Comm(tComm_list, VComm, comms);
    if (!Vertify_CnclComm(VComm, comms, VDev))
    {
        perror("Vertify_CnclComm");
        exit(EXIT_FAILURE);
    }
    if (!print_buffer_info(7, 256, VDev[7].get_send_buffer(), VDev[7].get_recv_buffer()))
    {
        perror("print_buffer_info error");
        exit(EXIT_FAILURE);
    }

    CNCL_CHECK(cnclDestroyComms(tComm_list, comms));
    std::this_thread::sleep_for(std::chrono::seconds(60));
    return 0;
}
