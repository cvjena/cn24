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
  static void GEMM(
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
    const int ldC);
  
  static void GEMV(
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
    const int incY);
  
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
  
  static void SETSAMPLE(
    Tensor& A,
    const int smA,
    const datum value);
  
  static void SMS(
    const Tensor& source,
    Tensor& target);
  
  static void SMS2(
    const Tensor& source,
    Tensor& target);
  
  static void DOWN(
    const Tensor& source,
    Tensor& target,
    const int region_width,
    const int region_height,
    const datum target_factor);
  
  static void UP(
    const Tensor& source,
    Tensor& target,
    const int region_width,
    const int region_height,
    const datum target_factor);
  
  static void ADD(
    const Tensor& source_a,
    const Tensor& source_b,
    Tensor& target);
};
  
}

#endif