#!/bin/bash

# Execute this file on the host
docker build -t cn24pkg .

TMP_BU=$(pwd)
cd ../..
CN24_SRC=$(pwd)
cd "$TMP_BU"
docker run -it --name cn24pkgbuilder -v $CN24_SRC:/cn24-pkg-out/cn24-src cn24pkg bash /cn24-pkg-out/build-package.sh
docker rm cn24pkgbuilder
