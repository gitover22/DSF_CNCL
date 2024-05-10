/***********************************************************************************
 * author : huafeng
 ***********************************************************************************/

#include <cncl.h>
#include <cnrt.h>
#include <thread> 
#include <vector>
#include <chrono>
#include <iostream>
#include "mluTool.h" // 需要配置.vscode文件添加自己的头文件搜索目录
int main(int argc, char* argv[]) {
    uint32_t num_dev = 0;
    GetMluNums(&num_dev);  
    std::cout<<"num of mlu: "<<num_dev<<std::endl;
    int *dev_list = new int[num_dev];
    // 初试化dev_list
    for (int i = 0; i < (int)num_dev; i++)
    {
        dev_list[i] = i;
    }
    int buf_count = (1<<28); // 缓冲区中包含的float类型变量的个数
    int buf_size = buf_count * sizeof(float); // buf_size为4MB
    std::unique_ptr<void*[]> send_buffer(new void*[num_dev]);
    std::unique_ptr<void*[]> recv_buffer(new void*[num_dev]);
    for(int i=0;i<num_dev;i++){
        CNRT_CHECK_TMP(cnrtSetDevice(dev_list[i]));
        CNRT_CHECK_TMP(cnrtMalloc(&send_buffer[i],buf_size)); // 一个mlu设备应该分配一个1GB的接受缓冲区和发送缓冲区 共2GB,每个设备的存储占用应该是2048MB
        CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer[i],buf_size));
        CNRT_CHECK_TMP(cnrtMemset(send_buffer[i],1,buf_size));
        CNRT_CHECK_TMP(cnrtMemset(recv_buffer[i],0,buf_size));
    }
    std::cout<<"allocate memory success please use cnmon to vertify"<<std::endl;
    // 调用chrono提供的休眠函数sleep 5秒 使用cnmon查看mlu卡的分配情况
    std::this_thread::sleep_for(std::chrono::seconds(50));
    // free
    for (int i = 0; i < num_dev; i++)
    {
        CNRT_CHECK_TMP(cnrtFree(send_buffer[i]));
        CNRT_CHECK_TMP(cnrtFree(recv_buffer[i]));
    }
    return 0;
}