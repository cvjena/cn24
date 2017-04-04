#!/bin/bash
if [ ! -e clblas ]; then
  wget https://github.com/clMathLibraries/clBLAS/releases/download/v2.12/clBLAS-2.12.0-Windows-x64.zip
  unzip clBLAS-2.12.0-Windows-x64.zip
  rm clBLAS-2.12.0-Windows-x64.zip
  mv package clblas
fi
