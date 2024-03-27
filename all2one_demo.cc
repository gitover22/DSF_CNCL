/***********************************************************************************
 * Copyright (c) 2018-2022, CAMBRICON TECHNOLOGIES CORPORATION. All rights reserved.
 ***********************************************************************************/

#include <cncl.h>
#include <cnrt.h>
#include <thread>  // NOLINT
#include <vector>

#include "check.h"  // NOLINT

int main(int argc, char* argv[]) {
  int num_comms = 4;
  int* dev_list = new int[num_comms];
  int* rank_list = new int[num_comms];
  cnclComm_t* comms = new cnclComm_t[num_comms];
  cnrtQueue_t* queues = new cnrtQueue_t[num_comms];
  void** send_buffer = new void*[num_comms];
  void** recv_buffer = new void*[num_comms];

  uint32_t num_dev = 0;
  CNRT_CHECK_TMP(cnrtGetDeviceCount(&num_dev));
  for (int i = 0; i < num_comms; i++) {
    rank_list[i] = i;  // comm's rank
    dev_list[i] = rank_list[i] % num_dev;
  }

  // allocate mlu buffers and queues
  int receiver = 0;
  int buf_count = (1 << 20);
  int buf_size = buf_count * sizeof(float);
  for (int i = 0; i < num_comms; i++) {
    CNRT_CHECK_TMP(cnrtSetDevice(dev_list[i]));
    CNRT_CHECK_TMP(cnrtQueueCreate(&queues[i]));
    CNRT_CHECK_TMP(cnrtMalloc(&send_buffer[i], buf_size));
    CNRT_CHECK_TMP(cnrtMemset(send_buffer[i], 1, buf_size));
    if (i == receiver) {
      for (int j = 0; j < num_comms; j++) {
        CNRT_CHECK_TMP(cnrtMalloc(&recv_buffer[j], buf_size));
        CNRT_CHECK_TMP(cnrtMemset(recv_buffer[j], 0, buf_size));
      }
    }
  }

  // initialize CNCL
  int nrank = num_comms;
  CNCL_CHECK(cnclInitComms(comms, num_comms, dev_list, rank_list, nrank, nullptr));

  // to do all2one
  std::vector<std::thread> threads;
  for (int i = 0; i < num_comms; i++) {
    threads.push_back(std::thread([=]() {
      CNCL_CHECK(cnclSend(
          send_buffer[i], buf_count, cnclFloat32, receiver, comms[i], queues[i]));
      if (i == receiver) {
        for (int j = 0; j < num_comms; j++) {
          CNCL_CHECK(cnclRecv(
              recv_buffer[j], buf_count, cnclFloat32, j, comms[i], queues[i]));
        }
      }
    }));
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
