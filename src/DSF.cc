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
int main(int argc , char *argv[])
{   
    uint32_t local_mlu_num = 0; // mlu数量
    GetMluNums(&local_mlu_num);
    printf("local mlu nums: %d\n",local_mlu_num);
    int comms; // 通信子数量
    printf("please input comm numbers (less or equal to mlu number): ");
    scanf("%d",&comms);
    getchar();
    Dev_MLU dev_list[comms];
    CNCLComm comm_list[comms];
    cnclComm_t *tComm_list = new cnclComm_t[comms];    // communicator list
    int *tDev_list = new int[comms];
    int *tRank_list = new int[comms];
    MapRankandDev(comms,local_mlu_num,tDev_list,tRank_list);

    for (int i = 0; i < comms; i++)
    {
        comm_list[i].init_rank(i);
        dev_list[i].init_device_id(comm_list[i].get_rank()%local_mlu_num);
        printf("please input mlu %d recvBuffer size, sendBuffer size(GB):",dev_list[i].get_device_id());
        int send_size=0,recv_size=0;
        scanf("%d %d",&send_size,&recv_size);
        dev_list[i].init_sendBuffer(send_size);
        dev_list[i].init_recvBuffer(recv_size);
        dev_list[i].init_queue();

    }
    cnclCliqueId clique_id;
    CNCL_CHECK(cnclGetCliqueId(&clique_id)); // 如何通过通信域id获取通信域内的通信子数量？
    // 初始化通信子
    InitComm(tComm_list,comms,tDev_list,tRank_list,comms,&clique_id);

    Map_Comm(tComm_list,comm_list,comms);
    CNCL_CHECK(cnclDestroyComms(tComm_list,comms));

    print_buffer_info(0,4,dev_list[0].get_send_buffer(),dev_list[0].get_recv_buffer());

    
    return 0;
}




