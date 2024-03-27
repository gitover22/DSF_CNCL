/***********************************************************************************
 * Copyright (c) 2018-2022, CAMBRICON TECHNOLOGIES CORPORATION. All rights reserved.
 ***********************************************************************************/

#include <cncl.h>
#include <cnrt.h>
#include <thread>  // NOLINT
#include <vector>

#include "check.h"  // NOLINT

int main(int argc, char* argv[]) {
  int num_comms = 4; // 通信子个数
  int* dev_list = new int[num_comms];
  int* rank_list = new int[num_comms];
  cnclComm_t* comms = new cnclComm_t[num_comms];
  cnrtQueue_t* queues = new cnrtQueue_t[num_comms];
  void** send_buffer = new void*[num_comms];
  void** recv_buffer = new void*[num_comms];

  uint32_t num_dev = 0; // 该系统中的设备数量
  CNRT_CHECK_TMP(cnrtGetDeviceCount(&num_dev)); //CNRT_CHECK_TMP:用来检查是否执行成功  cnrtGetDeviceCount:检索当前系统中MLU设备的数量,并返回给num_dev。
  for (int i = 0; i < num_comms; i++) {
    rank_list[i] = i;  // comm's rank
    dev_list[i] = rank_list[i] % num_dev;  // dev_list[i] = [0,1,2,3] % 8     dev_list = [0,1,2,3]
  }

  // allocate mlu buffers and queues
  int buf_count = (1 << 20);
  int buf_size = buf_count * sizeof(float); // 2^20 * 4
  for (int i = 0; i < num_comms; i++) {
    CNRT_CHECK_TMP(cnrtSetDevice(dev_list[i])); // cnrtSetDevice: 为调用宿主线程设置当前使用的MLU设备为“dev_list[i]”。
    CNRT_CHECK_TMP(cnrtQueueCreate(&queues[i])); // 在当前设备创建一个队列，返回队列指针给形参
    CNRT_CHECK_TMP(cnrtMalloc(&send_buffer[i], buf_size)); // cnrtMalloc：在当前设备上分配大小为buf_size的存储空间，返回指向该空间的指针给send_buffer
    CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer[i], buf_size));
    CNRT_CHECK_TMP(cnrtMemset(send_buffer[i], 1, buf_size));// cnrtMemset: 初试化空间的值
    CNRT_CHECK_TMP(cnrtMemset(recv_buffer[i], 0, buf_size));
  }

  // initialize CNCL
  int nrank = num_comms;
  CNCL_CHECK(cnclInitComms(comms, num_comms, dev_list, rank_list, nrank, nullptr)); // 在多进程间进行通信时初始化通信器。

  // to do allreduce
  std::vector<std::thread> threads;
  for (int i = 0; i < num_comms; i++) {
    std::thread t([=]() {
      CNCL_CHECK(cnclAllReduce(send_buffer[i],
                               recv_buffer[i],
                               buf_count,
                               cnclFloat32,
                               cnclSum,
                               comms[i],
                               queues[i]));
    });
    threads.push_back(std::move(t));
  }
  for (auto& t : threads) t.join();

  // wait for async tasks to complete
  for (int i = 0; i < num_comms; i++) {
    CNRT_CHECK_TMP(cnrtQueueSync(queues[i]));
  }

  // finalize CNCL
  CNCL_CHECK(cnclDestroyComms(comms, num_comms));

  // free mlu buffers and queues
  for (int i = 0; i < num_comms; i++) {
    CNRT_CHECK_TMP(cnrtQueueDestroy(queues[i]));
    CNRT_CHECK_TMP(cnrtFree(send_buffer[i]));
    CNRT_CHECK_TMP(cnrtFree(recv_buffer[i]));
  }

  delete[] queues;
  delete[] send_buffer;
  delete[] recv_buffer;
  delete[] comms;
  delete[] dev_list;
  delete[] rank_list;

  printf("Cncl runs in %d comms on %u devices: success.\n", num_comms, num_dev);
  return 0;
}
