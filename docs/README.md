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
    |-- cnclComm.cc
    |-- cnclComm.h
    |-- dev_MLU.cc
    |-- dev_MLU.h
    |-- DSF.cc
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
