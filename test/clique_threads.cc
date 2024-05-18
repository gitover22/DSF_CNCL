/**
 * @brief 多线程中的通信域
*/

#include <cncl.h>
#include <cnrt.h>
#include <thread> 
#include <vector>
#include <iostream>
#include <chrono>
#include <unistd.h>
#include <sys/types.h>
#include "mluTool.h"

void thread1_func(){
    int num_comms = 4;
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
    int count;
    CNCL_CHECK(cnclGetCommCount(&count, comms[0]));
    std::cout<<"comm    count: "<<count<<std::endl;
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
    std::cout<<"thread :"<<std::this_thread::get_id()<<std::endl;
    printf("Cncl runs in %d comms on %u devices: success.\n",num_comms, num_dev);
    std::this_thread::sleep_for(std::chrono::seconds(1000));

}
void thread2_func(){
    int num_comms = 8;
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
    int count;
    CNCL_CHECK(cnclGetCommCount(&count, comms[0]));
    std::cout<<"count: "<<count<<std::endl;


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
    std::cout<<"thread :"<<std::this_thread::get_id()<<std::endl;
    printf("Cncl runs in %d comms on %u devices: success.\n",num_comms, num_dev);
    std::this_thread::sleep_for(std::chrono::seconds(1000));



}
int main()
{
    cnclCliqueId clique_id;
    CNCL_CHECK(cnclGetCliqueId(&clique_id));
    std::thread t1(thread1_func);
    std::thread t2(thread2_func);

    t1.join();
    t2.join();

    return 0;
}