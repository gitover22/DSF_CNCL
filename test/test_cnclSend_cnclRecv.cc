/**
 * 实现mlu之间交换数据
 * @author huafeng
*/

#include <cncl.h>
#include <cnrt.h>
#include <thread>
#include <mluTool.h>
#include <vector>
#include <iostream>
#include <chrono>
#define NUM_OF_COMM 8
#define BUFFER_COUNT 4
void print_buffer_info(void** send_buffer,void** recv_buffer);
void operator_buffer_data(int m,int n,void **send_buffer,void **recv_buffer,cnclComm_t *comms,cnrtQueue_t *queues);
int main(int argc,char*argv[]){
    uint num_of_mlu ;
    GetMluNums(&num_of_mlu);
    int* rank = new int[NUM_OF_COMM];
    int* dev = new int[NUM_OF_COMM];
    cnclComm_t *comms = new cnclComm_t[NUM_OF_COMM];
    cnrtQueue_t *queues=new cnrtQueue_t[NUM_OF_COMM];
    void** send_buffer = new void*[NUM_OF_COMM];
    void** recv_buffer = new void*[NUM_OF_COMM];
    for(int i=0;i<NUM_OF_COMM;i++){
        rank[i] = i;
        dev[i] = rank[i] % num_of_mlu;
    }
    int* host_buffer = new int[BUFFER_COUNT];
    for(int i=0;i<NUM_OF_COMM;i++){
        cnrtSetDevice(dev[i]);
        cnrtQueueCreate(&queues[i]);
        cnrtMalloc(&send_buffer[i],BUFFER_COUNT * sizeof(int));
        cnrtMalloc(&recv_buffer[i],BUFFER_COUNT * sizeof(int));
        std::fill_n(host_buffer,BUFFER_COUNT,i);
        cnrtMemcpy(send_buffer[i],(void *)host_buffer,BUFFER_COUNT*sizeof(int),cnrtMemcpyHostToDev);
        std::fill_n(host_buffer,BUFFER_COUNT,NUM_OF_COMM-1-i);
        cnrtMemcpy(recv_buffer[i],(void *)host_buffer,BUFFER_COUNT*sizeof(int),cnrtMemcpyHostToDev);
    }
    cnclInitComms(comms,NUM_OF_COMM,dev,rank,NUM_OF_COMM,nullptr);
    print_buffer_info(send_buffer,recv_buffer);
    while(1){
        int m,n;
        printf("请输入想要发送数据的的mlu设备下标:");
        scanf("%d",&m);
        printf("请输入想要接受数据的的mlu设备下标:");
        scanf("%d",&n);
        operator_buffer_data(m,n,send_buffer,recv_buffer,comms,queues);
        printf("交换完成,交换后的缓冲区信息如下");
        print_buffer_info(send_buffer,recv_buffer);
        printf("请按回车键继续\n");
        getchar();
    }
    delete[] rank;
    delete[] dev;
    for (int i = 0; i < NUM_OF_COMM; i++)
    {
        CNRT_CHECK_TMP(cnrtQueueSync(queues[i]));
    }
    CNCL_CHECK(cnclDestroyComms(comms,NUM_OF_COMM));
    for (int i = 0; i <NUM_OF_COMM; i++)
    {
        CNRT_CHECK_TMP(cnrtQueueDestroy(queues[i]));
        CNRT_CHECK_TMP(cnrtFree(send_buffer[i]));
        CNRT_CHECK_TMP(cnrtFree(recv_buffer[i]));
    }
    return 0;
}
/**
 * @brief 打印缓冲区信息
 * @param send_buffer [in] 发送缓冲区
 * @param recv_buffer [in] 接收缓冲区
*/
void print_buffer_info(void** send_buffer,void** recv_buffer){
    for(int i=0;i<NUM_OF_COMM;i++){
        std::cout<<"mlu["<<i<<"] buffer data:"<<std::endl;
        int* buff_ptr = new int[BUFFER_COUNT];
        CNRT_CHECK_TMP(cnrtMemcpy((void *)buff_ptr, send_buffer[i], BUFFER_COUNT * sizeof(int), cnrtMemcpyDevToHost));
        std::cout<<"send_buffer: ";
        for(int j=0;j<BUFFER_COUNT;j++)
            std::cout << buff_ptr[j] << " ";
        std::cout<<std::endl;
        CNRT_CHECK_TMP(cnrtMemcpy((void *)buff_ptr, recv_buffer[i], BUFFER_COUNT * sizeof(int), cnrtMemcpyDevToHost));
        std::cout<<"recv_buffer: ";
        for(int j=0;j<BUFFER_COUNT;j++)
            std::cout << *(buff_ptr + j) << " ";
        std::cout<<std::endl;
        std::cout<<std::endl<<std::endl;
    }

}
/**
 * @brief 交换缓冲区数据
 * @param m [in] 发送数据的mlu下标
 * @param n [in] 接受数据的mlu下标
 * @param send_buffer [in] 所有设备的发送缓冲区
 * @param recv_buffer [in] 所有设备的接收缓冲区
*/
void operator_buffer_data(int m,int n,void **send_buffer,void **recv_buffer,cnclComm_t *comms,cnrtQueue_t *queues){
    CNCL_CHECK(cnclSend(send_buffer[m],BUFFER_COUNT,cnclInt32,n,comms[m],queues[m]));
    CNCL_CHECK(cnclRecv(recv_buffer[n],BUFFER_COUNT,cnclInt32,m,comms[n],queues[n]));
}
