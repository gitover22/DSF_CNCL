#ifndef DEV_MLU_H
#define DEV_MLU_H

#include "mluTool.h"
class Dev_MLU {
public:
    /**
     * @brief 初试化当前设备为id上的资源
     * @param id [in] 设备id号
     * @param send_buffer_size [in] 发送缓冲区的大小
     * @param recv_buffer_size [in] 接受缓冲区的大小
     * @param need_queue [in] 是否需要申请队列
    */
    Dev_MLU(int id, size_t send_buffer_size,size_t recv_bufffer_size,bool need_queue);

    /**
     * @brief 初试化发送缓冲区
    */
    int init_sendBuffer();

    /**
     * @brief 初试化接受缓冲区
    */
    int init_recvBuffer();

    /**
     * @brief 析构函数，释放缓冲区和队列
    */
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