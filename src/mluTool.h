#ifndef TOOL_H
#define TOOL_H

#include<string>
#include<iostream>
#include<vector>
#include<cnrt.h>
#include<cncl.h>
#include<memory>


using intPtr =std::unique_ptr<int[]>;
using cnclCommPtr = std::unique_ptr<cnclComm_t[]>;
using cnrtQueuePtr = std::unique_ptr<cnrtQueue_t[]>;
/**
 * @brief check CNRT result
*/
#define CNRT_CHECK_TMP(call)                                       \
  do {                                                             \
    cnrtRet_t ret_code = (call);                                   \
    if (ret_code != CNRT_RET_SUCCESS) {                            \
      printf("Runtime error, msg: %s", cnrtGetErrorStr(ret_code)); \
      exit(1);                                                     \
    }                                                              \
  } while (0)

/**
 * @brief get device nums
 * @param [out] nums pointer to the number of devices
*/
void GetMluNums(uint32_t *nums);

/**
 * @brief eazy map for dev and rank number
 * @param [in] num_comms number of comms
 * @param [in] num_dev number of mlu devices
 * @param [out] dev_list 设备队列的数组 
 * @param [out] rank_list 通信子对应的rank号数组
*/
void MapRankandDev(int num_comms,uint32_t num_dev,int *dev_list,int *rank_list);







void PrintList(int * list,int num);







#endif