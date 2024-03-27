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
  std::vector<cnrtQueue_t> queues(num_comms, nullptr);
  std::vector<void**> send_buffer(num_comms, nullptr);
  std::vector<void**> recv_buffer(num_comms, nullptr);
  std::vector<uint32_t*> sendbuf_cnts(num_comms, nullptr);
  std::vector<uint32_t*> recvbuf_cnts(num_comms, nullptr);
  cnclDataType_t* types = new cnclDataType_t[num_comms];

  uint32_t num_dev = 0;
  CNRT_CHECK_TMP(cnrtGetDeviceCount(&num_dev));
  for (int i = 0; i < num_comms; i++) {
    rank_list[i] = i;  // comm's rank
    dev_list[i] = rank_list[i] % num_dev;
  }

  // allocate mlu buffers and queues
  uint32_t buf_count = (1 << 20);
  uint32_t buf_size = buf_count * sizeof(float);
  for (int i = 0; i < num_comms; i++) {
    CNRT_CHECK_TMP(cnrtSetDevice(dev_list[i]));
    CNRT_CHECK_TMP(cnrtQueueCreate(&queues[i]));

    void** input = new void*[num_comms];
    void** output = new void*[num_comms];
    uint32_t* send_num = new uint32_t[num_comms];
    uint32_t* recv_num = new uint32_t[num_comms];
    for (int j = 0; j < num_comms; j++) {
      CNRT_CHECK_TMP(cnrtMalloc(&input[j], buf_size));
      CNRT_CHECK_TMP(cnrtMalloc(&output[j], buf_size));
      CNRT_CHECK_TMP(cnrtMemset(input[j], i, buf_size));
      CNRT_CHECK_TMP(cnrtMemset(output[j], 0, buf_size));
      send_num[j] = buf_count;
      recv_num[j] = buf_count;
    }
    send_buffer[i] = input;
    recv_buffer[i] = output;
    sendbuf_cnts[i] = send_num;
    recvbuf_cnts[i] = recv_num;
    types[i] = cnclFloat32;
  }

  // initialize CNCL
  int nrank = num_comms;
  CNCL_CHECK(cnclInitComms(comms, num_comms, dev_list, rank_list, nrank, nullptr));

  // to do all2all
  std::vector<std::thread> threads;
  for (int i = 0; i < num_comms; i++) {
    threads.push_back(std::thread([=]() {
      CNCL_CHECK(cnclAlltoAllv((const void**)send_buffer[i],
                               sendbuf_cnts[i],
                               types,
                               recv_buffer[i],
                               recvbuf_cnts[i],
                               types,
                               comms[i],
                               queues[i]));
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
    for (int j = 0; j < num_comms; j++) {
      CNRT_CHECK_TMP(cnrtFree(send_buffer[i][j]));
      CNRT_CHECK_TMP(cnrtFree(recv_buffer[i][j]));
    }
    delete[] send_buffer[i];
    delete[] recv_buffer[i];
    delete[] sendbuf_cnts[i];
    delete[] recvbuf_cnts[i];
  }

  delete[] comms;
  delete[] dev_list;
  delete[] rank_list;
  delete[] types;

  printf("Cncl runs in %d comms on %u devices: success.\n", num_comms, num_dev);
  return 0;
}
