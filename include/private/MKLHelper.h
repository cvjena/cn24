/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file MKLHelper.h
 * @brief Contains macros to help with various BLAS implementations.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_MKLHELPER_H
#define CONV_MKLHELPER_H
#ifdef BUILD_BLAS

/**
 * Here we include our favorite BLAS library and define important functions
 * so that they are independent of the type
 */

#ifdef BLAS_ACCELERATE
#include <Accelerate/Accelerate.h>
#define INNERGEMM cblas_sgemm
#define INNERGEMV cblas_sgemv

#endif

#ifdef BLAS_ATLAS
extern "C" {
#include <cblas.h>
}

#define INNERGEMM cblas_sgemm
#define INNERGEMV cblas_sgemv

#endif

#ifdef BLAS_ACML
extern "C" {
#include <cblas.h>
}

#define INNERGEMM cblas_sgemm
#define INNERGEMV cblas_sgemv

#endif

#ifdef BLAS_MKL

extern "C" {
#include <mkl_cblas.h>
}

#define INNERGEMM cblas_sgemm
#define INNERGEMV cblas_sgemv

#endif


#else

#include "Log.h"

#define INNERGEMM(...) { FATAL("GEMM function called without BLAS support!"); }
#define INNERGEMV(...) { FATAL("GEMV function called without BLAS support!"); }

#endif

#endif
