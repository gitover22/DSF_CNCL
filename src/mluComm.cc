/**
 @brife test for mlu comm
 @author huafeng
*/
#include "mluTool.h"
#include <memory>
#include <thread>
#include <iostream>

int main(int argc , char *argv[])
{   
    if(argc != 2){
        std::cout<<"Usage error! "<<std::endl \
        <<"Usage: ./test num_comms"<<std::endl;
        return 0;
    }
    int num_comms = atoi(argv[1]); // 一个comm对应一个mlu设备
    intPtr dev_list(new int[num_comms]);
    intPtr rank_list(new int[num_comms]); // 通信子的rank号  rank号用来标记comm

    cnclCommPtr comms(new cnclComm_t[num_comms]);
    cnrtQueuePtr queues(new cnrtQueue_t[num_comms]);
    std::unique_ptr<void*[]> send_buffer(new void*[num_comms]);
    std::unique_ptr<void*[]> recv_buffer(new void*[num_comms]);

    uint32_t num_dev = 0;
    GetMluNums(&num_dev);
    MapRankandDev(num_comms,num_dev,dev_list.get(),rank_list.get());

    int buf_count = (1<<20);
    int buf_size = buf_count * sizeof(float);
    // 对每个通信子进行分配空间等
    for(int i=0;i<num_comms;i++){
        CNRT_CHECK_TMP(cnrtSetDevice(dev_list[i]));
        CNRT_CHECK_TMP(cnrtQueueCreate(&queues[i])); // 在当前设备上创建一个队列
        CNRT_CHECK_TMP(cnrtMalloc(&send_buffer[i],buf_size)); // 在当前设备分配发送区缓存,\
        因为dev_list[i]可能被重复申请，所以这种场景下，一个设备可能有多个sendBuffer
        CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer[i],buf_size));
        CNRT_CHECK_TMP(cnrtMemset(send_buffer[i],1,buf_size));
        CNRT_CHECK_TMP(cnrtMemset(recv_buffer[i],0,buf_size));
    }
    CNCL_CHECK(cnclInitComms(comms.get(),num_comms,dev_list.get(),rank_list.get(),num_comms,nullptr));
    std::vector<std::thread> threads;
    for(int i=0;i<num_comms;i++){
        // lamda表达式 [=]会导致拷贝所有外部作用域中的变量，包括std::unique_ptr。由于std::unique_ptr禁止拷贝（只能移动）\
        所以这里在[]中声明所需引用的变量和需要拷贝的变量 
        std::thread t([&send_buffer, &recv_buffer, buf_count, i, &comms, &queues](){
            CNCL_CHECK(cnclAllReduce(send_buffer[i],recv_buffer[i],buf_count,cnclFloat32,cnclSum,comms[i],queues[i]));
        });
        threads.push_back(std::move(t)); // 因为tread资源不允许拷贝构造，所以只能转化为右值，利用右值引用的移动构造来转移资源所有权
    }
    for(auto& t:threads) t.join();
    for (int i = 0; i < num_comms; i++)
    {
        CNRT_CHECK_TMP(cnrtQueueSync(queues[i]));
    }

    CNCL_CHECK(cnclDestroyComms(comms.get(),num_comms));
    for (int i = 0; i < num_comms; i++)
    {
        CNRT_CHECK_TMP(cnrtQueueDestroy(queues[i]));
        CNRT_CHECK_TMP(cnrtFree(send_buffer[i]));
        CNRT_CHECK_TMP(cnrtFree(recv_buffer[i]));

    }
    printf("Cncl runs in %d comms on %u devices: success.\n", num_comms, num_dev);
    
    
    return 0;
}