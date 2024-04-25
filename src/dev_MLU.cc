#include "dev_MLU.h"

// 构造函数：初始化设备、队列、缓冲区
Dev_MLU::Dev_MLU(int id, size_t send_buffer_size,size_t recv_bufffer_size):device_id(id), \
                    send_buffer_size(send_buffer_size),recv_buffer_size(recv_buffer_size) {
                        
    
}

// 对外接口：进行通信操作
void Dev_MLU::performCommunication(cnclComm_t comm) {
    
}

// 同步队列
void Dev_MLU::syncQueue() {
    CNRT_CHECK_TMP(cnrtQueueSync(queue));
}

// 析构函数：清理资源
Dev_MLU::~Dev_MLU() {
    CNRT_CHECK_TMP(cnrtQueueDestroy(queue));
    CNRT_CHECK_TMP(cnrtFree(send_buffer));
    CNRT_CHECK_TMP(cnrtFree(recv_buffer));
}
