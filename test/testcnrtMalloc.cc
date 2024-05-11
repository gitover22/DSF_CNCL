/***********************************************************************************
 * author : huafeng
 * date   : 2024-05
 * desc   : test cnrtMalloc
 ***********************************************************************************/

#include <cncl.h>
#include <cnrt.h>
#include <thread> 
#include <vector>
#include <chrono>
#include <iostream>
#include "mluTool.h"
int main(int argc, char* argv[]) {
    uint32_t num_dev = 0;
    GetMluNums(&num_dev);  
    std::cout<<"num of mlu: "<<num_dev<<std::endl;
    int *dev_list = new int[num_dev];
    for (int i = 0; i < (int)num_dev; i++)
    {
        dev_list[i] = i;
    }
    int buf_count = (1<<28); 
    int buf_size = buf_count * sizeof(float);
    std::unique_ptr<void*[]> send_buffer(new void*[num_dev]);
    std::unique_ptr<void*[]> recv_buffer(new void*[num_dev]);
    for(int i=0;i<num_dev;i++){
        CNRT_CHECK_TMP(cnrtSetDevice(dev_list[i]));
        CNRT_CHECK_TMP(cnrtMalloc(&send_buffer[i],buf_size)); // one mlu should be allocated about 2048MB
        CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer[i],buf_size));
        CNRT_CHECK_TMP(cnrtMemset(send_buffer[i],1,buf_size));
        CNRT_CHECK_TMP(cnrtMemset(recv_buffer[i],0,buf_size));
    }
    std::cout<<"allocate memory success please use cnmon to vertify"<<std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(50));
    // free
    for (int i = 0; i < num_dev; i++)
    {
        CNRT_CHECK_TMP(cnrtFree(send_buffer[i]));
        CNRT_CHECK_TMP(cnrtFree(recv_buffer[i]));
    }
    return 0;
}