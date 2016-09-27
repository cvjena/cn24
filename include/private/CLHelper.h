/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file CLHelper.h
 * @brief Provides OpenCL utility functions
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CLHELPER_H
#define CONV_CLHELPER_H

#ifdef BUILD_OPENCL
#ifdef __APPLE__
#include <cl.h>
#else
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#include <CL/cl.h>
#endif

#ifdef BUILD_CLBLAS
#include <clBLAS.h>
#endif
#endif

#include "Init.h"

namespace Conv {
class CLHelper {
public:
  static void Init(unsigned int platform_number = 0,
                   unsigned int device_number = 0);
#ifdef BUILD_OPENCL
  static cl_program CreateProgram(const char* file_name);
  static long bytes_up;
  static long bytes_down;
  static cl_device_id device;
  static cl_context context;
  static cl_command_queue queue;
  
  // Kernels
  static cl_kernel k_crossCorrelation;
  static cl_kernel k_biasedConvolution;
  static cl_kernel k_fullConvolution;
  static cl_kernel k_biasedMatrixVector;
  static cl_kernel k_biasedMatrixVectorGrad;
  static cl_kernel k_biasedMatrixVectorBackward;
  static cl_kernel k_biasGradientPart1;
  static cl_kernel k_biasGradientPart2;
  static cl_kernel k_matrixMatrix;
  static cl_kernel k_foldWeights;
  static cl_kernel k_maximumForward;
  static cl_kernel k_maximumBackward;
  static cl_kernel k_amaximumForward;
  static cl_kernel k_amaximumBackward;
  static cl_kernel k_nlTanh;
  static cl_kernel k_nlTanhBackward;
  static cl_kernel k_nlSigm;
  static cl_kernel k_nlSigmBackward;
  static cl_kernel k_nlLeaky;
  static cl_kernel k_nlLeakyBackward;
  static cl_kernel k_setValue;
  static cl_kernel k_sms;
  static cl_kernel k_im2col;
  static cl_kernel k_col2im;
  static cl_kernel k_up;
  static cl_kernel k_down;
  static cl_kernel k_applyMask;
#endif
};
  
}

#endif
