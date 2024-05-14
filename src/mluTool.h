#ifndef TOOL_H
#define TOOL_H

#include<string>
#include<iostream>
#include<vector>
#include<cnrt.h>
#include<cncl.h>
#include<memory>

// 智能指针
using intPtr =std::unique_ptr<int[]>;
using cnclCommPtr = std::unique_ptr<cnclComm_t[]>;
using cnrtQueuePtr = std::unique_ptr<cnrtQueue_t[]>;

/**
 * @brief 检查CNRT系列函数执行结果
*/
#define CNRT_CHECK_TMP(call)                                       \
  do {                                                             \
    cnrtRet_t ret_code = (call);                                   \
    if (ret_code != CNRT_RET_SUCCESS) {                            \
      printf("Runtime error, msg: %s", cnrtGetErrorStr(ret_code)); \
      exit(1);                                                     \
    }                                                              \
  } while (0)

#define CNCL_CHECK(cmd)                                              \
  do {                                                                    \
    cnclResult_t error = cmd;                                             \
    if (error != CNCL_RET_SUCCESS) {                                      \
      std::string err = "CNCL error in: " + std::string(__FILE__) + ":" + \
                        std::to_string(__LINE__) + ", " +                 \
                        std::string(cnclGetErrorStr(error));              \
    }                                                                     \
  } while (0)

#define CNCL_ASSERT(cmd)                                           \
  do {                                                                  \
    cnclResult_t res = cmd;                                             \
    if (res != CNCL_RET_SUCCESS) {                                      \
      std::string err = cnclGetErrorStr(res);                           \
      fprintf(stderr, "CNCL error in: %s:%d, %s\n", __FILE__, __LINE__, \
              err.c_str());                                             \
      abort();                                                          \
    }                                                                   \
  } while (0)

/**
 * @brief 获取设备数量
 * @param nums [out] 接受指向设备数量的地址
*/
void GetMluNums(uint32_t *nums);

/**
 * @brief 映射rank号数组和设备数组
 * @param num_comms [in] 通信子数量
 * @param num_dev [in] 设备数量
 * @param dev_list [out] 设备队列的数组 
 * @param rank_list [out] 通信子对应的rank号数组
*/
void MapRankandDev(int num_comms,uint32_t num_dev,int *dev_list,int *rank_list);

/**
 * @brief 打印int型数组
 * @param list [in] 待打印数组的地址
 * @param num [in] 数组长度
*/
void PrintList(int * list,int num);

#endif