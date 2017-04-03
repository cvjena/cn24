#!/bin/bash
CN24_VERSION="3.0.0-SNAPSHOT"
cd "${0%/*}"

# Create "origin" tarball
cd cn24-src
tar -cz -f ../cn24_$CN24_VERSION.orig.tar.gz *
cd ..

mkdir cn24-$CN24_VERSION
cd cn24-$CN24_VERSION
tar -x -f ../cn24_$CN24_VERSION.orig.tar.gz
mv distribution/debian debian

dpkg-buildpackage -uc -us -j12

bash
