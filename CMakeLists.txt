cmake_minimum_required(VERSION 2.8)
set(CMAKE_C_COMPILER /usr/bin/clang)
set(CMAKE_CXX_COMPILER /usr/bin/clang++)
include_directories("$ENV{NEUWARE_HOME}/include")
include_directories("$ENV{CMAKE_CURRENT_SOURCE_DIR}/..")
link_directories("$ENV{NEUWARE_HOME}/lib64") # 添加链接库的搜索路径
add_definitions("-g -Wall -std=c++11") # 编译器选项
# GLOB参数告诉CMake去查找匹配指定模式的文件，并将这些文件的列表存储在CNCL_SAMPLES_SRCS变量中
file(GLOB CNCL_SAMPLES_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

foreach(sample ${CNCL_SAMPLES_SRCS})
  get_filename_component(fname ${sample} NAME) # 带扩展名：broadcast_demo.cc
  get_filename_component(fname_we ${sample} NAME_WE) # 提取不带扩展名的文件名broadcast_demo
  add_executable(${fname_we} ${fname})
  target_link_libraries(${fname_we} cncl cnrt cndrv cndev pthread rt)
endforeach()
