#include "mluTool.h"
#include <memory>
#include <thread>
#include <iostream>
/**
 @brife test for mlu comm
 @author huafeng
*/
int main(int argc , char *argv[])
{   
    if(argc != 2){
        std::cout<<"Usage error! "<<std::endl \
        <<"Usage: ./test num_comms"<<std::endl;
        return 0;
    }
    int num_comms = atoi(argv[1]);
    std::unique_ptr<int[]> dev_list(new int(num_comms)); // 设备
    std::unique_ptr<int[]> rank_list(new int(num_comms)); // 通信子的rank号 ，用来标记comm？？
    // 两个结构体指针数组
    cnclComm_t* comms = new cnclComm_t[num_comms];
    cnrtQueue_t* queues = new cnrtQueue_t[num_comms];

    // 创建发送缓冲区智能指针数组
    std::unique_ptr<void*[]> send_buffer(new void*[num_comms]);

    // 创建接收缓冲区智能指针数组
    std::unique_ptr<void*[]> recv_buffer(new void*[num_comms]);

    uint32_t num_dev = 0;  // 当前机子上的mlu设备数量
    GetMluNums(&num_dev);
    MapRankandDev(num_comms,num_dev,dev_list.get(),rank_list.get());
    // for(int i=0;i<num_comms;i++){
    //     printf("dev_list[%d]: %d\t rank_list[%d]: %d\n",i,dev_list[i],i,rank_list[i]);  
    // }
    return 0;
}