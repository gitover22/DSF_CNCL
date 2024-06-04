#include "dev_MLU.h"
#include "mluTool.h"
// 构造函数：初始化设备、队列、缓冲区
Dev_MLU::Dev_MLU(int id, int send_buffer_size, int recv_buffer_size, bool need_queue) : device_id(id),
                                                                                        send_buffer_size(send_buffer_size), recv_buffer_size(recv_buffer_size)
{
    if (need_queue)
        init_queue();
    init_sendBuffer(send_buffer_size);
    init_recvBuffer(recv_buffer_size);
}

void Dev_MLU::init_sendBuffer(int s_KB)
{
    this->send_buffer_size = s_KB * (2 << 10);
    // this->send_buffer_size = 4*sizeof(int); // test
    CNRT_CHECK_TMP(cnrtSetDevice(device_id));
    CNRT_CHECK_TMP(cnrtMalloc(&send_buffer, send_buffer_size));
    CNRT_CHECK_TMP(cnrtMemset(send_buffer, 0, send_buffer_size));
}

void Dev_MLU::init_recvBuffer(int r_KB)
{
    this->recv_buffer_size = r_KB * (2 << 10);
    // this->recv_buffer_size = 4*sizeof(int); // test
    int *host_buffer = new int[1024 / 4];
    std::fill_n(host_buffer, 1024 / 4, this->device_id);
    // printf("device:%d \n", this->device_id);
    CNRT_CHECK_TMP(cnrtSetDevice(device_id));
    CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer, recv_buffer_size));
    cnrtMemcpy(recv_buffer, (void *)host_buffer, 256 * sizeof(int), cnrtMemcpyHostToDev);

    // CNRT_CHECK_TMP(cnrtMemset(recv_buffer,0,recv_buffer_size));
}

Dev_MLU::Dev_MLU(Dev_MLU &&other) noexcept
    : device_id(other.device_id), send_buffer_size(other.send_buffer_size), recv_buffer_size(other.recv_buffer_size),
      send_buffer(other.send_buffer), recv_buffer(other.recv_buffer), queue(other.queue)
{
    other.device_id = -1;
    other.send_buffer_size = 0;
    other.recv_buffer_size = 0;
    other.send_buffer = nullptr;
    other.recv_buffer = nullptr;
    other.queue = nullptr;
}

Dev_MLU &Dev_MLU::operator=(Dev_MLU &&other) noexcept
{
    if (this != &other)
    {
        // Clean up the existing resources
        CNRT_CHECK_TMP(cnrtQueueSync(queue));
        CNRT_CHECK_TMP(cnrtFree(send_buffer));
        CNRT_CHECK_TMP(cnrtFree(recv_buffer));
        CNRT_CHECK_TMP(cnrtQueueDestroy(queue));
        
        // Transfer ownership
        device_id = other.device_id;
        send_buffer_size = other.send_buffer_size;
        recv_buffer_size = other.recv_buffer_size;
        send_buffer = other.send_buffer;
        recv_buffer = other.recv_buffer;
        queue = other.queue;

        // Reset the moved-from object
        other.device_id = -1;
        other.send_buffer_size = 0;
        other.recv_buffer_size = 0;
        other.send_buffer = nullptr;
        other.recv_buffer = nullptr;
        other.queue = nullptr;
    }
    return *this;
}

Dev_MLU::~Dev_MLU()
{
    // printf("dev[%d] 析构 \n", device_id);
    if (send_buffer)
    {
        CNRT_CHECK_TMP(cnrtQueueSync(queue));
        CNRT_CHECK_TMP(cnrtFree(send_buffer));
    }
    if (recv_buffer)
    {
        CNRT_CHECK_TMP(cnrtFree(recv_buffer));
    }
    if (queue)
    {
        CNRT_CHECK_TMP(cnrtQueueDestroy(queue));
    }
}

bool Dev_MLU::init_device_id(int device_id)
{
    this->device_id = device_id;
    return true;
}

bool Dev_MLU::init_queue()
{
    CNRT_CHECK_TMP(cnrtSetDevice(device_id));
    CNRT_CHECK_TMP(cnrtQueueCreate(&queue));
    return true;
}

int Dev_MLU::get_device_id()
{
    return device_id;
}

void *Dev_MLU::get_send_buffer()
{
    // std::cout << "mlu[" << this->device_id << "] buffer data:" << std::endl;
    // int *buff_ptr = new int[256];
    // CNRT_CHECK_TMP(cnrtMemcpy((void *)buff_ptr, send_buffer, 256 * sizeof(int), cnrtMemcpyDevToHost));
    // std::cout << "send_buffer: ";
    // for (int j = 0; j < 256; j++)
    //     std::cout << buff_ptr[j] << " ";
    // std::cout << std::endl;
    // std::cout << send_buffer<<std::endl;
    return (void *)send_buffer;
}

void *Dev_MLU::get_recv_buffer()
{
    return recv_buffer;
}