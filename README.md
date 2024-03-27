快速入门
=================

CNCL样例
-------------

开发样例目录结构

```
|-- samples
    |-- build.sh
    |-- CMakeLists.txt
    |-- broadcast_demo.cc
    |-- allgather_demo.cc
    |-- allreduce_demo.cc
    |-- reduce_demo.cc
    |-- reducescatter_demo.cc
    |-- one2all_demo.cc
    |-- all2one_demo.cc
    |-- all2all_demo.cc
```

样例所有文件介绍

samples文件夹

- CMakeLists.txt：cmake描述文件，用于编译样例。
- build.sh: 首先检查依赖库文件是否完整，然后使用cmake编译所有样例。
- broadcast_demo.cc：cnclBroadcast接口使用样例。
- allgather_demo.cc : cnclAllGather接口使用样例。
- allreduce_demo.cc : cnclAllReduce接口使用样例。
- reduce_demo.cc : cnclReduce接口使用样例。
- reducescatter_demo.cc : cnclReduceScatter接口使用样例。
- one2all_demo.cc : cnclSend、cnclRecv接口使用样例。
- all2one_demo.cc : cnclSend、cnclRecv接口使用样例。
- all2all_demo.cc : cnclAlltoAllv接口使用样例。


依赖库目录结构

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

运行样例：

```
#按照上述依赖库目录结构，创建文件夹存放编译样例所需依赖的头文件和动态库
export NEUWARE_HOME=/path/to/your/neuware_home
export LD_LIBRARY_PATH=${NEUWARE_HOME}/lib64:${LD_LIBRARY_PATH}
./build.sh
cd build
./broadcast_demo
./allreduce_demo
```
