#include "mluTool.h"
#include <sys/types.h>

void GetMluNums(uint32_t *nums){
    CNRT_CHECK_TMP(cnrtGetDeviceCount(nums));
    std::cout<<"------------------------MLU numbers: "<<*nums<<"------------------------"<<std::endl;
}

void MapRankandDev(int num_comms,uint32_t num_dev,int *dev_list,int *rank_list){
    for (int i = 0; i < num_comms; i++) {
        rank_list[i] = i;  // comm's rank
        dev_list[i] = rank_list[i] % num_dev;  // dev_list[i] = [0,1,2,3] % 8     dev_list = [0,1,2,3]
    }
}

void PrintList(int *list,int num){
// 打印dev_list，rank_list
    for(int i=0;i<num;i++){
        printf("list[%d]: %d\n",i,list[i]);  
    }
}