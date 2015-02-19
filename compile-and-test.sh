#!/bin/sh

echo "-- CN24 Demo for Urban Scene Understanding"
echo "--- CN24 Compilation"
mkdir -p build
cd build/
cmake -DCN24_BUILD_JPG:BOOL=ON ..
make
if [ $? -ne 0 ]; then
    echo "--- CN24 Compilation failed, please compile manually (see wiki)"
    exit -1
fi
echo "--- CN24 Test run"
sh ../scripts/download_examples.sh
time ./classifyImage ../example/labelmefacade.set ../example/labelmefacade.net ../example/lmf_pretrained.Tensor sample3.jpg output3.jpg
echo "--- CN24 Input image: build/sample3.jpg"
echo "--- CN24 Result image: build/output3.jpg"
