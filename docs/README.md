DSF_CNCL
=================

项目简介
- DSF_CNCL：DSM System Framework based on CNCL
- 基于mlu370集群，CNCL搭建DSM框架
- 项目未完成，待做！！



目录结构

```
|-- docs
    |-- README.md
|-- src
    |-- build.sh
    |-- CMakeLists.txt
    |-- dev_MLU.cc
    |-- dev_MLU.h
    |-- mluComm.cc
    |-- mluTool.cc
    |-- mluTool.h
|-- test
    |-- build.sh
    |-- CMakeLists.txt
    |-- test_cnclAllReduce.cc
    |-- test_cnclSend_cnclRecv.cc
    |-- test_cnrtMalloc.cc
    |-- test_MLU.cc
|-- .gitignore
|-- LICENSE
|-- push.sh
```

文件介绍

src

- CMakeLists.txt：cmake描述文件，用于编译样例。
- build.sh: 检查依赖库文件是否完整，并创建build，使用cmake编译所有样例。
- Makefile: 简易构建项目，用于简单自测
- mluComm.cc: 主函数，待做
- mluTool.cc: 工具实现
- mluTool.h: 工具定义

依赖库目录结构(cambricon官网可下载)

```
|-- neuware_home
|   |-- include
|   |   |-- cncl.h
|   |   |-- cnrt.h
|   |-- lib64
|   |   |-- libcndev.so
|   |   |-- libcndrv.so
|   |   |-- libcnbin.so
|   |   |-- libcnrt.so
|   |   |-- libcncl.so
|     
```

RUN：
需要有mlu370的集群环境，具体配置可参考[官方手册](https://www.cambricon.com/docs/sdk_1.15.0/cncl_1.13.0/user_guide/4_quickstart/installation/index.html)

```
cd src
./build.sh
cd build
./mluComm
```
