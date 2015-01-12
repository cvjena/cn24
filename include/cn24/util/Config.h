/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file Config.h
 * \brief Contains configuration macros.
 * 
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_CONFIG_H
#define CONV_CONFIG_H


#include <sys/types.h>
#include <cstdint>

namespace Conv {
  
//define BRUTAL_FINISH
  
/**
 * This makes the networks data type configurable without using
 * templates. Templates do not allow for move constructors and
 * are evil.
 */
typedef float datum;
typedef int32_t dint;
#ifdef __MINGW32__
typedef uint32_t duint;
#else
#ifdef _MSC_VER
typedef uint32_t duint;
#else
typedef u_int32_t duint;
#endif
#endif
#define DATUM_FROM_UCHAR(x) ((Conv::datum)(0.003921569f * ((unsigned char)x)))
#define UCHAR_FROM_DATUM(x) ((unsigned char) (255.0f * ((Conv::datum)x) ) )
#define MCHAR_FROM_DATUM(x) ((unsigned char) (127.0f + 127.0f * ((Conv::datum)x) ) )

}

#ifdef BUILD_BLAS
/**
 * Here we include our favorite BLAS library and define important functions
 * so that they are independent of the type
 */
#ifdef BLAS_ATLAS
extern "C" {
#include <cblas.h>
}

#define GEMM cblas_sgemm

#endif

#ifdef BLAS_ACML
extern "C" {
#include <cblas.h>
}

#define GEMM cblas_sgemm

#endif

#ifdef BLAS_MKL

extern "C" {
#include <mkl_cblas.h>
}

#define GEMM cblas_sgemm

#endif


#else

#include "Log.h"

#define GEMM(...) { FATAL("GEMM function called without BLAS support!"); }

#endif


#endif