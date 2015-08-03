/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file TensorMath.h
 * @class TensorMath
 * @brief Global functions for math operations using Tensors
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CONV_TENSORMATH_H
#define CONV_TENSORMATH_H

#include "../util/Log.h"
#include "../util/Config.h"
#include "../util/Tensor.h"

namespace Conv {

class TensorMath {
public:
  // Datum version of functions
  static void RAW_GEMM(
    const bool is_row_major,
    const bool transpose_A,
    const bool transpose_B,
    const int M,
    const int N,
    const int K,
    const datum alpha,
    const datum* A,
    const int ldA,
    const datum* B,
    const int ldB,
    const datum beta,
    datum* C,
    const int ldC);
  
  static void RAW_GEMV(
    const bool is_row_major,
    const bool transpose_A,
    const int M,
    const int N,
    const datum alpha,
    const datum* A,
    const int ldA,
    const datum* X,
    const int incX,
    const datum beta,
    datum* Y,
    const int incY);
  
  // Fancy tensor versions of functions
  inline static void GEMM(
    const bool is_row_major,
    const bool transpose_A,
    const bool transpose_B,
    const int M,
    const int N,
    const int K,
    const datum alpha,
    const Tensor& A,
    const int smA,
    const int ldA,
    const Tensor& B,
    const int smB,
    const int ldB,
    const datum beta,
    Tensor& C,
    const int smC,
    const int ldC)
  {
    RAW_GEMM(is_row_major, transpose_A, transpose_B, M, N, K, alpha, A.data_ptr_const(0,0,0,smA), ldA, B.data_ptr_const(0,0,0,smB), ldB, beta, C.data_ptr(0,0,0,smC), ldC);
  }
  
  inline static void GEMV(
    const bool is_row_major,
    const bool transpose_A,
    const int M,
    const int N,
    const datum alpha,
    const Tensor& A,
    const int smA,
    const int ldA,
    const Tensor& X,
    const int smX,
    const int incX,
    const datum beta,
    Tensor& Y,
    const int smY,
    const int incY)
  {
    RAW_GEMV(is_row_major, transpose_A, M, N, alpha, A.data_ptr_const(0,0,0,smA), ldA, X.data_ptr_const(0,0,0,smX),incX,beta,Y.data_ptr(0,0,0,smY),incY);
  }
  
  static void IM2COL(
    const Tensor& source,
    const int source_width,
    const int source_height,
    const int maps,
    const int samples,
    const int kernel_width,
    const int kernel_height,
    const int stride_width,
    const int stride_height,
    const int pad_width,
    const int pad_height,
    Tensor& target);
  
  static void COL2IM(
    Tensor& source,
    const int source_width,
    const int source_height,
    const int maps,
    const int samples,
    const int kernel_width,
    const int kernel_height,
    const int stride_width,
    const int stride_height,
    const int pad_width,
    const int pad_height,
    const Tensor& target);
};
  
}

#endif