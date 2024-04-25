/**
 @brife test for mlu comm
 @author huafeng
*/
#include "mluTool.h"
#include <memory>
#include <thread>
#include <iostream>
/**
 * 一个通信子对应一个设备，通信子用rank号来进行标记
*/
int main(int argc , char *argv[])
{   
    if(argc != 2){
        std::cout<<"Usage error! "<<std::endl \
        <<"Usage: ./runner num_comms"<<std::endl;
        return 0;
    }
    int test_mlu(int);
    test_mlu(atoi(argv[1]));
    return 0;
}








