#!/bin/bash
git clone https://github.com/clMathLibraries/clBLAS.git -b v2.6
cd clBLAS
mkdir build
cd build
cmake -DBUILD_TEST:BOOL=OFF ../src
sudo make install
