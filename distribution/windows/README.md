# Prerequisites

- AMD APP SDK (even for NVIDIA users)
  - Install at least "OpenCL Runtime".

- Intel OpenCL driver for Xeon/Core
  - Install only if you don't have a gpu driver

- MSYS 2 (using version 20161025)
  - Follow MSYS's installation instructions including running `pacman -Syu` etc.
  - Install `mingw-w64-x86_64-gcc`, `mingw-w64-x86_64-cmake`, `mingw-w64-x86_64-readline`, `p7zip` and `make`.

- Inno Setup (using version 5.5.9)
  - Also install Inno Download Plugin

# Building the setup package

- Start MSYS2 MinGW 64
- Navigate to `distribution/windows/msys` folder
- Execute `./build-packages.sh`
- Check log for errors :)
- Run Inno Script Studio with `distribution/windows/inno/cn24.iss`
- Compile
