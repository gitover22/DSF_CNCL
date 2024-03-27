cmake_minimum_required(VERSION 2.8)
project(samples C CXX)

include_directories("$ENV{NEUWARE_HOME}/include")
include_directories("$ENV{CMAKE_CURRENT_SOURCE_DIR}/..")
link_directories("$ENV{NEUWARE_HOME}/lib64")
add_definitions("-g -Wall -std=c++11")

file(GLOB CNCL_SAMPLES_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/*.cc
)

foreach(sample ${CNCL_SAMPLES_SRCS})
  get_filename_component(fname ${sample} NAME)
  get_filename_component(fname_we ${sample} NAME_WE)
  add_executable(${fname_we} ${fname})
  target_link_libraries(${fname_we} cncl cnrt cndrv cndev pthread rt)
endforeach()
