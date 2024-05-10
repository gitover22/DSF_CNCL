/**
* @brief test for mlu comm
* @author huafeng
*/
#include "mluTool.h"
#include "dev_MLU.h"
#include <memory>
#include <thread>
#include <iostream>
/**
 * 一个通信子对应一个设备，通信子用rank号来进行标记
*/
int main(int argc , char *argv[])
{   
    uint32_t local_mlu_num = 0; // mlu数量
    GetMluNums(&local_mlu_num);
    int comms; // 通信子数量
    printf("please input comm numbers (less or equal to mlu number): ");
    scanf("%d",&comms);
    getchar();
    

    return 0;
}




