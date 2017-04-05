#!/bin/bash

# This script runs inside the container
CN24_VERSION="3.0.0-SNAPSHOT"
cd "${0%/*}"

echo "Making target directory"
mkdir cn24-$CN24_VERSION
cd cn24-$CN24_VERSION

# Copy files from repo
echo "Copying files..."
shopt -s extglob
cp -r ../cn24-src/!(distribution) ./

echo "Moving debian control files..."
cp -r ../cn24-src/distribution/ubuntu-deb/debian debian

# Create "origin" tarball
echo "Creating origin tarball..."
tar -cz -f ../cn24_$CN24_VERSION.orig.tar.gz *

echo "Building package..."
debuild -us -uc -j12
bash
