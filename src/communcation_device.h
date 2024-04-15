#ifndef COMMUNCATION_DEVICE_H
#define COMMUNCATION_DEVICE_H

#include "mluTool.h"
class CommunicationDevice {
public:
    // 构造函数：初始化设备、队列、缓冲区
    CommunicationDevice(int id, size_t buf_size);

    // 对外接口：进行通信操作
    void performCommunication(cnclComm_t comm);

    // 同步队列
    void syncQueue();

    // 析构函数：清理资源
    ~CommunicationDevice();

private:
    int device_id;
    cnrtQueue_t queue;
    void* send_buffer;
    void* recv_buffer;
    size_t buffer_size;
};


#endif