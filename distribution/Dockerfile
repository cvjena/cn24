FROM ubuntu:16.04
MAINTAINER Clemens-Alexander Brust (clemens-alexander.brust@uni-jena.de)

# Install development packages
RUN apt-get update && \
  apt-get install -y \
  git-core \
  cmake \
  opencl-headers \
  ocl-icd-libopencl1 \
  ocl-icd-opencl-dev \
  libclblas-dev \
  libpng12-dev \
  libjpeg-dev \
  libpng12-0 \
  libjpeg8 \
  libreadline6-dev \
  packaging-dev

RUN mkdir /cn24-pkg-out
COPY scripts/build-package.sh /cn24-pkg-out/build-package.sh
