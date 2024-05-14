/**
* @brief test for mlu comm
* @author huafeng
*/
#include "mluTool.h"
#include "dev_MLU.h"
#include "cnclComm.h"
#include <memory>
#include <thread>
#include <iostream>
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
    for (int i = 0; i < comms; i++)
    {
        comm_list[i].init_rank(i);
        dev_list[i].init_device_id(comm_list[i].get_rank()%local_mlu_num);
        printf("请输入mlu %d的发送缓存大小,接收缓存大小(GB):",dev_list[i].get_device_id());
        int send_size=0,recv_size=0;
        // scanf("%d %d",&send_size,&recv_size);
        dev_list[i].init_sendBuffer(send_size);
        dev_list[i].init_recvBuffer(recv_size);
        dev_list[i].init_queue();
        // int clique_total_comm = get_clique_totalComm(comm_list[i].getCnclComm());
        cnclCliqueId clique_id;
        CNCL_CHECK(cnclGetCliqueId(&clique_id)); // 如何通过通信域id获取通信域内的通信子数量？
        // comm_list[i].init_comm(clique_total_comm,comm_list[i].get_rank(),dev_list[i].get_device_id(),clique_id);
        printf("=== debug1 ===\n");
        comm_list[i].init_comm(comms,dev_list[i].get_device_id(),comm_list[i].get_rank(),&clique_id);

        printf("=== debug1 ===\n");
    }
    print_buffer_info(0,4,dev_list[0].get_send_buffer(),dev_list[0].get_recv_buffer());

    
    return 0;
}




