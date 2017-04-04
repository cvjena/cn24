#!/bin/sh
mkdir -p ../dist-tmp/cpu
mkdir -p ../dist-tmp/gpu
mkdir -p ../dist-tmp/gpu-cv
mkdir -p build-cpu
mkdir -p build-gpu
mkdir -p build-gpu-cv

#   # (1) CPU BUILD
#   cd build-cpu
#   cmake -DJPEG_INCLUDE_DIR=$(pwd)/../libjpeg/include -DJPEG_LIBRARY=$(pwd)/../libjpeg/lib/libjpeg.a -DCMAKE_BUILD_TYPE=Release -G "MSYS Makefiles" ../../../..
#   make -j12
#   ctest . --rerun-failed
#   cp -r libcn24.dll libcn24.dll.a cn24-shell.exe ../../dist-tmp/cpu/
#   cd ..
#   
#   # (2) GPU BUILD
#   cd ../packages
#   ./download-clblas.sh
#   cd ../msys
#   cd build-gpu
#   cmake -DJPEG_INCLUDE_DIR=$(pwd)/../libjpeg/include -DJPEG_LIBRARY=$(pwd)/../libjpeg/lib/libjpeg.a -DCMAKE_BUILD_TYPE=Release -DCN24_BUILD_OPENCL=ON -G "MSYS Makefiles" -DCN24_BUILD_OPENCL_CLBLAS=ON -DCLBLAS_INCLUDE_DIR=../../packages/clblas/include -DCLBLAS_LIBRARY=../../packages/clblas/bin/clBLAS.dll ../../../..
#   make -j12
#   PATH=$PATH:../../packages/clblas/bin ctest . --rerun-failed
#   cp -r libcn24.dll libcn24.dll.a cn24-shell.exe ../../dist-tmp/gpu/
#   cd ..

# (3) GPU+CV BUILD
cd ../packages
./download-opencv.sh
cd ../msys
cd build-gpu-cv
OpenCV_DIR=../../packages/opencv/build2 cmake -DJPEG_INCLUDE_DIR=$(pwd)/../libjpeg/include -DJPEG_LIBRARY=$(pwd)/../libjpeg/lib/libjpeg.a -DCMAKE_BUILD_TYPE=Release -DCN24_BUILD_OPENCL=ON -G "MSYS Makefiles" -DCN24_BUILD_OPENCL_CLBLAS=ON -DCLBLAS_INCLUDE_DIR=../../packages/clblas/include -DCLBLAS_LIBRARY=../../packages/clblas/bin/clBLAS.dll -DCN24_BUILD_OPENCV=ON ../../../..
make -j12
PATH=$PATH:../../packages/clblas/bin ctest . --rerun-failed
cp -r libcn24.dll libcn24.dll.a cn24-shell.exe ../../dist-tmp/gpu-cv/
cd ..


