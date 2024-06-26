#! /bin/bash

# 检查neuware_home是否设置
check_neuware_home_path=true
# 检查lib64
check_lib64=true
# 检查include
check_include=true
# 检查环境变量NEUWARE_HOME是否存在  | -z: 判断字符串是否为空 | !: 取反 | -f: 文件是否存在
if [ ! -z "${NEUWARE_HOME}" ]; then
    echo "-- using NEUWARE_HOME = ${NEUWARE_HOME}"

    echo "[ CHECKING   ] NEUWARE_HOME lib64/..."
    if [ ! -f "${NEUWARE_HOME}/lib64/libcndev.so" ]; then
        check_lib64=false
        echo "-- missing libcndev.so in NEUWARE_HOME env."
    fi
    if [ ! -f "${NEUWARE_HOME}/lib64/libcndrv.so" ]; then
        check_lib64=false
        echo "-- missing libcndrv.so in NEUWARE_HOME env."
    fi
    if [ ! -f "${NEUWARE_HOME}/lib64/libcnbin.so" ]; then
        check_lib64=false
        echo "-- missing libcnbin.so in NEUWARE_HOME env."
    fi
    if [ ! -f "${NEUWARE_HOME}/lib64/libcnrt.so" ]; then
        check_lib64=false
        echo "-- missing libcnrt.so in NEUWARE_HOME env."
    fi
    if [ ! -f "${NEUWARE_HOME}/lib64/libcncl.so" ]; then
        check_lib64=false
        echo "-- missing libcncl.so in NEUWARE_HOME env."
    fi
    if [ "$check_lib64" == true ]; then
        echo "[     PASSED ] NEUWARE_HOME lib64/..."
    else
        echo "[     FAILED ] NEUWARE_HOME lib64/..."
    fi

    echo "[ CHECKING   ] NEUWARE_HOME include/..."
    if [ ! -f "${NEUWARE_HOME}/include/cnrt.h" ]; then
        check_include=false
        echo "-- missing cnrt.h in NEUWARE_HOME env."
    fi
    if [ ! -f "${NEUWARE_HOME}/include/cncl.h" ]; then
        check_include=false
        echo "-- missing cncl.h in NEUWARE_HOME env."
    fi
    if [ "$check_include" == true ]; then
        echo "[     PASSED ] NEUWARE_HOME include/..."
    else
        echo "[     FAILED ] NEUWARE_HOME include/..."
    fi

else
    check_neuware_home_path=false
    echo "-- please prepare NEUWARE_HOME env following README.md."
fi
if [ "$check_lib64" == false ] || [ "$check_include" == false ] || [ "$check_neuware_home_path" == false ]; then
    exit -1
fi

set -e
BUILD_DIR="build"
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
# pushd popd 入栈出栈 like cd
pushd ${BUILD_DIR}
    cmake ..
    make
popd
