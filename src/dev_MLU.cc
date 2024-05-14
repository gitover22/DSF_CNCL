#include "dev_MLU.h"
#include "mluTool.h"
// 构造函数：初始化设备、队列、缓冲区
Dev_MLU::Dev_MLU(int id, int send_buffer_size,int recv_buffer_size, bool need_queue):device_id(id), \
                    send_buffer_size(send_buffer_size),recv_buffer_size(recv_buffer_size) {
        // 初始化设备上的队列
        if (need_queue)
            init_queue();
        init_sendBuffer(send_buffer_size);
        init_recvBuffer(recv_buffer_size);    
}

void Dev_MLU::init_sendBuffer(int s_GB){
    // this->send_buffer_size = s_GB * (2<<30);
    this->send_buffer_size = 4*sizeof(int); // test
    CNRT_CHECK_TMP(cnrtSetDevice(device_id));
    CNRT_CHECK_TMP(cnrtMalloc(&send_buffer,send_buffer_size));
    CNRT_CHECK_TMP(cnrtMemset(send_buffer,0,send_buffer_size));
}

void Dev_MLU::init_recvBuffer(int r_GB){
    // this->recv_buffer_size = r_GB * (2<<30);
    this->recv_buffer_size = 4*sizeof(int); // test
    CNRT_CHECK_TMP(cnrtSetDevice(device_id));
    CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer,recv_buffer_size));
    CNRT_CHECK_TMP(cnrtMemset(recv_buffer,0,recv_buffer_size));
}

Dev_MLU::~Dev_MLU() {
    CNRT_CHECK_TMP(cnrtQueueDestroy(queue));
    CNRT_CHECK_TMP(cnrtFree(send_buffer));
    CNRT_CHECK_TMP(cnrtFree(recv_buffer));
}

bool Dev_MLU::init_device_id(int device_id){
    this->device_id = device_id;
    return true;
}

bool Dev_MLU::init_queue(){
    CNRT_CHECK_TMP(cnrtSetDevice(device_id));
    CNRT_CHECK_TMP(cnrtQueueCreate(&queue));
    return true;
}

int Dev_MLU::get_device_id(){
    return device_id;
}

void *Dev_MLU::get_send_buffer(){
    return send_buffer;
}

void *Dev_MLU::get_recv_buffer(){
    return recv_buffer;
}