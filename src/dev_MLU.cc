#include "dev_MLU.h"
#include "mluTool.h"
// 构造函数：初始化设备、队列、缓冲区
Dev_MLU::Dev_MLU(int id, size_t send_buffer_size,size_t recv_bufffer_size, bool need_queue):device_id(id), \
                    send_buffer_size(send_buffer_size),recv_buffer_size(recv_buffer_size) {
        // 初始化设备上的队列
        if (need_queue)
            CNRT_CHECK_TMP(cnrtQueueCreate(&queue));
        CNRT_CHECK_TMP(cnrtSetDevice(device_id));
        init_sendBuffer();
        init_recvBuffer();    
}

int Dev_MLU::init_sendBuffer(){
    CNRT_CHECK_TMP(cnrtMalloc(&send_buffer,send_buffer_size));
    CNRT_CHECK_TMP(cnrtMemset(send_buffer,1,send_buffer_size));
}

int Dev_MLU::init_recvBuffer(){
    CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer,recv_buffer_size));
    CNRT_CHECK_TMP(cnrtMemset(recv_buffer,0,recv_buffer_size));
}

Dev_MLU::~Dev_MLU() {
    CNRT_CHECK_TMP(cnrtQueueDestroy(queue));
    CNRT_CHECK_TMP(cnrtFree(send_buffer));
    CNRT_CHECK_TMP(cnrtFree(recv_buffer));
}
