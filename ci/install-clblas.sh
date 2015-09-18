#!/bin/bash
git clone https://github.com/clMathLibraries/clBLAS.git
cd clBLAS
mkdir build
cd build
cmake -DBUILD_TEST:BOOL=OFF ../src
sudo make install
