cmake_minimum_required(VERSION 3.0)
project(Test)

# 设置编译器为 clang
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# 设置 C++ 标准和编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# 编译器选项
add_definitions("-g -Wall -std=c++11")

# 设置头文件搜索路径 cncl.h等
include_directories("/usr/local/neuware/include")
include_directories("/home/zouguoqiang/DSF_CNCL/src")

# 添加链接库的搜索路径  libcncl.so等搜索路径
link_directories("$ENV{NEUWARE_HOME}/lib64") 

# 添加可执行文件生成规则
add_executable(test_MLU test_MLU.cc ../src/mluTool.cc)
add_executable(test_cnrtMalloc test_cnrtMalloc.cc ../src/mluTool.cc)
add_executable(test_cnclAllReduce test_cnclAllReduce.cc ../src/mluTool.cc)
add_executable(test_cnclSend_cnclRecv test_cnclSend_cnclRecv.cc ../src/mluTool.cc)
add_executable(clique_threads clique_threads.cc ../src/mluTool.cc)

# 链接所需的库
target_link_libraries(test_MLU stdc++ cncl cnrt cndrv cndev pthread rt)
target_link_libraries(test_cnrtMalloc stdc++ cncl cnrt cndrv cndev pthread rt)
target_link_libraries(test_cnclAllReduce stdc++ cncl cnrt cndrv cndev pthread rt)
target_link_libraries(test_cnclSend_cnclRecv stdc++ cncl cnrt cndrv cndev pthread rt)
target_link_libraries(clique_threads stdc++ cncl cnrt cndrv cndev pthread rt)