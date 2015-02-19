#!/bin/sh

echo CN24 Compilation and Test Run
mkdir build
cd build/
cmake -DCN24_BUILD_JPG:BOOL=ON ..
make
sh ../scripts/download_examples.sh
./classifyImage ../example/labelmefacade.set ../example/labelmefacade.net ../example/lmf_pretrained.Tensor sample3.jpg output3.jpg
