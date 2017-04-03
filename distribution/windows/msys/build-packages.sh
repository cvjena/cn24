#!/bin/sh
mkdir -p ../dist-tmp
mkdir -p build
cd build
cmake -DJPEG_INCLUDE_DIR=$(pwd)/../libjpeg/include -DJPEG_LIBRARY=$(pwd)/../libjpeg/lib/libjpeg.a -DCMAKE_BUILD_TYPE=Release -G "MSYS Makefiles" ../../../..
make -j12
ctest .
cp -r libcn24.dll libcn24.dll.a cn24-shell.exe ../../dist-tmp/
cd ..
