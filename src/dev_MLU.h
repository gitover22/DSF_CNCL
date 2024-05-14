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
    Dev_MLU(int id, int send_buffer_size,int recv_bufffer_size,bool need_queue);
    /**
     * @brief 默认构造函数
    */
    Dev_MLU() = default;
    /**
     * @brief 初试化当前设备id号
    */
    bool init_device_id(int device_id);
    /**
     * @brief 获取当前设备id号
    */
    int get_device_id();
    /**
     * @brief 初试化队列
    */
    bool init_queue();   
    /**
     * @brief 初试化发送缓冲区
    */
    void init_sendBuffer(int s_GB);

    /**
     * @brief 初试化接受缓冲区
    */
    void init_recvBuffer(int r_GB);

    /**
     * @brief 析构函数，释放缓冲区和队列
    */
    ~Dev_MLU();
    
    void *get_send_buffer();
    void *get_recv_buffer();
private:
    int device_id; // 当前设备id
    cnrtQueue_t queue; // 该mlu上的队列
    void* send_buffer; // 该mlu指向发送缓冲区的指针
    void* recv_buffer; // 该mlu指向接受缓冲区的指针
    size_t send_buffer_size; // 该mlu上发送缓冲区大小
    size_t recv_buffer_size; // 该mlu上接受缓冲区大小
};


#endif