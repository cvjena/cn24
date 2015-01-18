# How to build CN24
First, make sure you have all the required dependencies. Then, clone the
CN24 repository:

```bash
git clone https://github.com/cvjena/cn24.git
```

Create a build directory and run CMake:

```bash
mkdir build
cmake path/to/cn24
```

Run your preferred build tool, for example:
```
make
```

That's it, you're done!

# Dependencies
CN24 uses the CMake cross-platform build system. You need at least version
2.8 to generate the build files.
The following compilers are supported for building CN24:
* GCC >= 4.8
* Clang >= 3.5
* Visual Studio >= 2013

Older versions will probably work as long as they support the C++11 features
used by CN24. All other dependencies are optional. Optional dependencies include:
* _libjpeg_ and _libpng_ to read .jpg and .png files
* _Intel MKL_, _AMD ACML_ or _ATLAS_ for faster calculations
* _OpenCL_ for GPU acceleration
* _GTK+ 3_ for GUI utilities
