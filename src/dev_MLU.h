#ifndef DEV_MLU_H
#define DEV_MLU_H

#include "mluTool.h"
class Dev_MLU {
public:
    // 初始化设备、队列、缓冲区
    Dev_MLU(int id, size_t send_buffer_size,size_t recv_bufffer_size);

    // 对外接口：进行通信操作
    void performCommunication(cnclComm_t comm);

    // 同步队列
    void syncQueue();

    // 释放资源
    ~Dev_MLU();
    

private:
    int device_id; // 当前设备id
    cnrtQueue_t queue; // 该mlu上的队列
    void* send_buffer; // 该mlu指向发送缓冲区的指针
    void* recv_buffer; // 该mlu指向接受缓冲区的指针
    size_t send_buffer_size; // 该mlu上发送缓冲区大小
    size_t recv_buffer_size; // 该mlu上接受缓冲区大小
};


#endif