/***********************************************************************************
 * author : huafeng
 * date   : 2024-05
 * desc   : test runtime environment
 ***********************************************************************************/

#include <cncl.h>
#include <cnrt.h>
#include <thread> 
#include <vector>
#include <iostream>
#include <chrono>
#include "mluTool.h"
int main(int argc, char* argv[]) {
    if(argc != 2){
        std::cout<<"Usage error! "<<std::endl \
        <<"Usage: ./runner num_comms"<<std::endl;
        return 0;
    }
    int test_mlu(int);
    test_mlu(atoi(argv[1]));
    return 0;
}

/**
 * @brief test for environment
 * @param comm_num [in] num of comms
*/
int test_mlu(int comm_num){
    int num_comms = comm_num;
    intPtr dev_list(new int[num_comms]);
    intPtr rank_list(new int[num_comms]);

    cnclCommPtr comms(new cnclComm_t[num_comms]);
    cnrtQueuePtr queues(new cnrtQueue_t[num_comms]);
    std::unique_ptr<void*[]> send_buffer(new void*[num_comms]);
    std::unique_ptr<void*[]> recv_buffer(new void*[num_comms]);

    uint32_t num_dev = 0;
    GetMluNums(&num_dev);
    MapRankandDev(num_comms,num_dev,dev_list.get(),rank_list.get());

    int buf_count = (1<<2);
    int buf_size = buf_count * sizeof(float);

    float* host_buffer = new float[buf_count];
    std::fill_n(host_buffer, buf_count, 1.0f);


    // allocate
    for(int i=0;i<num_comms;i++){
        CNRT_CHECK_TMP(cnrtSetDevice(dev_list[i]));
        CNRT_CHECK_TMP(cnrtQueueCreate(&queues[i]));
        CNRT_CHECK_TMP(cnrtMalloc(&send_buffer[i],buf_size)); 
        CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer[i],buf_size));

        CNRT_CHECK_TMP(cnrtMemset(send_buffer[i],1,buf_size));
        CNRT_CHECK_TMP(cnrtMemset(recv_buffer[i],0,buf_size));
    }
    CNCL_CHECK(cnclInitComms(comms.get(),num_comms,dev_list.get(),rank_list.get(),num_comms,nullptr));

    std::vector<std::thread> threads;
    for(int i=0;i<num_comms;i++){
        std::thread t([&send_buffer, &recv_buffer, buf_count, i, &comms, &queues](){
            CNCL_CHECK(cnclAllReduce(send_buffer[i],recv_buffer[i],buf_count,cnclFloat32,cnclSum,comms[i],queues[i]));
        });
        threads.push_back(std::move(t));
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