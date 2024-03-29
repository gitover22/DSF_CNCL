cmake_minimum_required(VERSION 3.0)
project(mlu_test)

# 设置编译器为 clang
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)

# 设置 C++ 标准和编译选项
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")

# 设置头文件搜索路径
include_directories("/usr/local/neuware/include")

# 添加可执行文件生成规则
add_executable(mlu_test mlu_comm.cc tool.cc)

# 链接所需的库
target_link_libraries(mlu_test stdc++)
