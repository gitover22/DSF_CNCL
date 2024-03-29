#include <cncl.h>
#include <cnrt.h>
#include "tool.h"
#include <memory>
#include <iostream>
/**
 @brife test for mlu comm
 @author huafeng
*/
int main(int argc , char *argv[])
{   
    if(argc != 2){
        std::cout<<"Usage error! "<<std::endl \
        <<"Usage: ./test num_comms"<<std::endl;
        return 0;
    }
    int num_comms = atoi(argv[1]);
    std::unique_ptr<int[]> dev_list(new int(num_comms));
    
    return 0;
}