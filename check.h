/***********************************************************************************
 * Copyright (c) 2018-2022, CAMBRICON TECHNOLOGIES CORPORATION. All rights reserved.
 ***********************************************************************************/

#ifndef SAMPLES_CHECK_H_
#define SAMPLES_CHECK_H_

#include <cncl.h>

/***********************************************************************************
 * Check CNRT result
 ***********************************************************************************/

#define CNRT_CHECK_TMP(call)                                       \
  do {                                                             \
    cnrtRet_t ret_code = (call);                                   \
    if (ret_code != CNRT_RET_SUCCESS) {                            \
      printf("Runtime error, msg: %s", cnrtGetErrorStr(ret_code)); \
      exit(1);                                                     \
    }                                                              \
  } while (0)

#endif  // SAMPLES_CHECK_H_
