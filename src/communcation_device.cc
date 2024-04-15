#include "communcation_device.h"

// 构造函数：初始化设备、队列、缓冲区
CommunicationDevice::CommunicationDevice(int id, size_t buf_size) : device_id(id), buffer_size(buf_size) {
    CNRT_CHECK_TMP(cnrtSetDevice(device_id));
    CNRT_CHECK_TMP(cnrtQueueCreate(&queue));
    CNRT_CHECK_TMP(cnrtMalloc(&send_buffer, buffer_size));
    CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer, buffer_size));
    CNRT_CHECK_TMP(cnrtMemset(send_buffer, 1, buffer_size));
    CNRT_CHECK_TMP(cnrtMemset(recv_buffer, 0, buffer_size));
}

// 对外接口：进行通信操作
void CommunicationDevice::performCommunication(cnclComm_t comm) {
    CNCL_CHECK(cnclAllReduce(send_buffer, recv_buffer, buffer_size / sizeof(float), cnclFloat32, cnclSum, comm, queue));
}

// 同步队列
void CommunicationDevice::syncQueue() {
    CNRT_CHECK_TMP(cnrtQueueSync(queue));
}

// 析构函数：清理资源
CommunicationDevice::~CommunicationDevice() {
    CNRT_CHECK_TMP(cnrtQueueDestroy(queue));
    CNRT_CHECK_TMP(cnrtFree(send_buffer));
    CNRT_CHECK_TMP(cnrtFree(recv_buffer));
}
