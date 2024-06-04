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
    for(int i=0;i<num;i++){
        printf("list[%d]: %d\n",i,list[i]);  
    }
}

int get_clique_totalComm(const cnclComm_t comm){
    int count;
    CNCL_CHECK(cnclGetCommCount(&count, comm));
    return count;
}

/**
 * @brief 打印缓冲区信息
 * @param send_buffer [in] 发送缓冲区
 * @param recv_buffer [in] 接收缓冲区
*/
bool Print_buffer_info(int dev_index,int buffer_count,void* send_buffer,void* recv_buffer){
    std::cout<<"mlu["<<dev_index<<"] buffer data:"<<std::endl;
    int* buff_ptr = new int[buffer_count];
    CNRT_CHECK_TMP(cnrtMemcpy((void *)buff_ptr, send_buffer, buffer_count * sizeof(int), cnrtMemcpyDevToHost));
    std::cout<<"send_buffer: ";
    for(int j=0;j<buffer_count;j++)
        std::cout << buff_ptr[j] << " " << std::flush;
    std::cout<<std::endl;

    CNRT_CHECK_TMP(cnrtMemcpy((void *)buff_ptr, recv_buffer, buffer_count * sizeof(int), cnrtMemcpyDevToHost));
    std::cout<<"recv_buffer: "  << std::flush;
    for(int j=0;j<buffer_count;j++){
        std::cout << buff_ptr[j] <<" " << std::flush;
    }
    std::cout<<std::endl;
    return true;

}
