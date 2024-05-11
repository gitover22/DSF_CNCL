/***********************************************************************************
 * author : huafeng
 ***********************************************************************************/

#include <cncl.h>
#include <cnrt.h>
#include <thread>
#include <mluTool.h>
#include <vector>
#include <iostream>
#include <chrono>
/**
 * vertify the correctness of cnclAllReduce
 * sendbuffer[0]:[0,0,0,0,0]  sendbuffer[1]:[1,1,1,1,1]  sendbuffer[2]:[2,2,2,2,2]  ...  sendbuffer[n]:[n,n,n,n,n]
 * after AllReduce :  recvbuffer is [0+1+2+...+n,0+1+2+...+n,0+1+2+...+n,0+1+2+...+n,0+1+2+...+n] 
*/
void AllReduce(int num_comms, int buf_count, void** send_buffer, void** recv_buffer, cnclComm_t* comms, cnrtQueue_t* queues);
int main(int argc, char* argv[]) {
    
    int num_of_comm = 8; // number of communicators
    int *dev_list = new int[num_of_comm];   // device list  dev_list[i] = x : comm i 's device is x
    int *rank_list = new int[num_of_comm];  // rank list    rank_list[i] = x : comm i 's rank is x
    cnclComm_t *comm_list = new cnclComm_t[num_of_comm];    // communicator list
    void **send_buffer = new void*[num_of_comm];   // send buffer
    void **recv_buffer = new void*[num_of_comm];   // recv buffer
    uint num_of_mlu;
    CNRT_CHECK_TMP(cnrtGetDeviceCount(&num_of_mlu));
    printf("num_of_mlu: %d\n", num_of_mlu);
    for (int i = 0; i < num_of_comm; i++)
    {
        rank_list[i] = i;
        dev_list[i] = rank_list[i] % num_of_mlu;
    }
    int buffer_count = 5; // every buffer elements number is 10
    // allocate buffer
    int * host = new int[buffer_count]; // local buffer
    for(int i = 0; i < num_of_comm; i++){
        CNRT_CHECK_TMP(cnrtSetDevice(dev_list[i]));
        // CNRT_CHECK_TMP(cnrtQueueCreate(&queue_list[i]));
        CNRT_CHECK_TMP(cnrtMalloc(&send_buffer[i], buffer_count * sizeof(int)));
        CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer[i], buffer_count * sizeof(int)));
        CNRT_CHECK_TMP(cnrtMemset(recv_buffer[i], 0, buffer_count * sizeof(int)));
        // init host
        std::fill_n(host, buffer_count, i);
        CNRT_CHECK_TMP(cnrtMemcpy(send_buffer[i], (void *)host, buffer_count * sizeof(int), cnrtMemcpyHostToDev));
    }
    // init comms
    CNCL_CHECK(cnclInitComms(comm_list,num_of_comm,dev_list,rank_list,num_of_comm,nullptr));
    
    AllReduce(num_of_comm, buffer_count, send_buffer, recv_buffer, comm_list, nullptr);

    // read recvbuffer's data
    for(int i=0;i<num_of_comm;i++){
        std::cout<<"recv_buffer["<<i<<"]: "<<std::endl;

        int* buff_ptr = new int[buffer_count];
        CNRT_CHECK_TMP(cnrtMemcpy((void *)buff_ptr, recv_buffer[i], buffer_count * sizeof(int), cnrtMemcpyDevToHost));
        // std::cout<<float_ptr<<std::endl;
        for(int j=0;j<buffer_count;j++)
            std::cout << *(buff_ptr + j) << " ";
        std::cout<<std::endl<<std::endl;
    }
    CNCL_CHECK(cnclDestroyComms(comm_list,num_of_comm));
    for (int i = 0; i < num_of_comm; i++)
    {
        CNRT_CHECK_TMP(cnrtFree(send_buffer[i]));
        CNRT_CHECK_TMP(cnrtFree(recv_buffer[i]));

    }
    delete[] dev_list;
    delete[] rank_list;
    delete[] comm_list;
    printf("Cncl runs in %d comms on %u devices: success.\n", num_of_comm, num_of_mlu);
    return 0;  
}
void AllReduce(int num_comms, int buf_count, void** send_buffer, void** recv_buffer, cnclComm_t* comms, cnrtQueue_t* queues){
 std::vector<std::thread> threads;
    for(int i=0;i< num_comms;i++){ 
        std::thread t([&send_buffer, &recv_buffer, buf_count, i, &comms, &queues](){
            CNCL_CHECK(cnclAllReduce(send_buffer[i],recv_buffer[i],buf_count,cnclInt32,cnclSum,comms[i],nullptr));
        });
        threads.push_back(std::move(t));
    }
    for(auto& t:threads) t.join(); 
}