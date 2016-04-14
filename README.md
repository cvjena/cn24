#### Build status:
master (production branch): [![Build Status](https://travis-ci.org/cvjena/cn24.svg?branch=master)](https://travis-ci.org/cvjena/cn24)
develop (development branch): [![Build Status](https://travis-ci.org/cvjena/cn24.svg?branch=develop)](https://travis-ci.org/cvjena/cn24)

## Welcome to the CN24 GitHub repository!

CN24 is a complete semantic segmentation framework using fully convolutional networks. It supports a wide variety of
platforms (Linux, Mac OS X and Windows) and libraries (OpenCL, Intel MKL, AMD ACML...) while providing dependency-free
reference implementations. The software is developed in the [Computer Vision Group](http://www.inf-cv.uni-jena.de) at the [University of Jena](http://www.uni-jena.de).

## Why should I use CN24?
1. Designed for *pixel-wise labeling and semantic segmentation* (train and test your own networks!)
2. Suited for *various applications* in [driver assistance systems](http://hera.inf-cv.uni-jena.de:6680/pdf/Brust15:CPN.pdf), scene understanding, remote sensing, biomedical image processing and many more
3. *OpenCL support* not only suited for NVIDIA GPUs
4. High-performance implementation with *minimal dependencies* to other libraries

## Getting started
To get started, clone this repository and visit the [wiki](https://github.com/cvjena/cn24/wiki)! Installation is just a two command lines away. For an even faster introduction, check out one of these examples:

* [Urban Scene Understanding Example](https://github.com/cvjena/cn24/wiki/Urban-Scene-Understanding-Example)
* [Road Detection Example](https://github.com/cvjena/cn24/wiki/Road-Detection-Example)

The repository contains pre-trained networks for these two applications, which are ready to use.

### Licensing
CN24 is available under a 3-clause BSD license. See [LICENSE](LICENSE) for details.
If you use CN24 for research, please cite our paper
[Clemens-Alexander Brust, Sven Sickert, Marcel Simon, Erik Rodner, Joachim Denzler. "Convolutional Patch Networks with Spatial Prior for Road Detection and Urban Scene Understanding". VISAPP 2015](http://hera.inf-cv.uni-jena.de:6680/pdf/Brust15:CPN.pdf).

Remark: The paper does not discuss the fully convolutional network adaptations integrated in CN24.

### Questions?
If you have questions, feedback, or experience problems. Let us know and write an e-mail to 
[Clemens-Alexander Brust](http://github.com/clrokr), [Sven Sickert](http://www.inf-cv.uni-jena.de/sickert), [Marcel Simon](http://www.inf-cv.uni-jena.de/simon), and [Erik Rodner](http://www.erodner.de).
