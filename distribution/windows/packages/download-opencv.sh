#/bin/bash

# (1) Download OpenCV
if [ ! -e opencv ]; then
  curl -o opencv.exe -L -O https://sourceforge.net/projects/opencvlibrary/files/opencv-win/3.2.0/opencv-3.2.0-vc14.exe/download
  7z x opencv.exe
  rm opencv.exe
fi

# (2) Compile OpenCV
if [ ! -e opencv/build2 ]; then
  cd opencv
  mkdir build2
  cd build2
  cmake -G "MSYS Makefiles" ../sources
  make -j12
fi
