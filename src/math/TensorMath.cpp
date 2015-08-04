/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "MKLHelper.h"
#include "CLHelper.h"

#include "TensorMath.h"

namespace Conv {
  
void TensorMath::GEMM(const bool is_row_major, const bool transpose_A, const bool transpose_B, const int M, const int N, const int K, const datum alpha, const Conv::Tensor &A, const int smA, const int ldA, const Conv::Tensor &B, const int smB, const int ldB, const datum beta, Conv::Tensor &C, const int smC, const int ldC)
{
#ifdef BUILD_CLBLAS
  ((Tensor&)A).MoveToGPU();
  ((Tensor&)B).MoveToGPU();
  C.MoveToGPU(C.hint_ignore_content_ && beta == 0.0);
  
  cl_event done_event = NULL;
  
  const int offA = A.width() * A.height() * A.maps() * smA;
  const int offB = B.width() * B.height() * B.maps() * smB;
  const int offC = C.width() * C.height() * C.maps() * smC;
  
  cl_int err =
    clblasSgemm(is_row_major ? clblasRowMajor : clblasColumnMajor,
    transpose_A ? clblasTrans : clblasNoTrans,
    transpose_B ? clblasTrans : clblasNoTrans,
    M, N, K, alpha, (cl_mem)A.cl_data_ptr_, offA, ldA,
    (cl_mem)B.cl_data_ptr_, offB, ldB, beta,
    (cl_mem)C.cl_data_ptr_, offC, ldC,
    1, &(CLHelper::queue), 0, NULL, &done_event);
  
  if(err!=CL_SUCCESS)
    FATAL("Call to clblasSgemm failed. Error: " << err);
  
  err = clWaitForEvents(1, &done_event);
  
  if(err!=CL_SUCCESS)
    FATAL("Call to clWaitForEvents failed. Error: " << err);
#else
#ifdef BUILD_BLAS
  INNERGEMM(is_row_major ? CblasRowMajor : CblasColMajor,
    transpose_A ? CblasTrans : CblasNoTrans,
    transpose_B ? CblasTrans : CblasNoTrans,
    M, N, K,
    alpha, A.data_ptr_const(0,0,0,smA), ldA,
    B.data_ptr_const(0,0,0,smB), ldB,
    beta, C.data_ptr(0,0,0,smC), ldC);
#else
  FATAL("No reference GEMM at this time!");
#endif // BUILD_BLAS
#endif // BUILD_CLBLAS
  C.hint_ignore_content_ = false;
}
  
void TensorMath::GEMV(const bool is_row_major, const bool transpose_A, const int M, const int N, const datum alpha, const Conv::Tensor &A, const int smA, const int ldA, const Conv::Tensor &X, const int smX, const int incX, const datum beta, Conv::Tensor &Y, const int smY, const int incY)
{
#ifdef BUILD_CLBLAS
  ((Tensor&)A).MoveToGPU();
  ((Tensor&)X).MoveToGPU();
  Y.MoveToGPU(Y.hint_ignore_content_ && beta == 0.0);
  
  cl_event done_event = NULL;
  
  const int offA = A.width() * A.height() * A.maps() * smA;
  const int offX = X.width() * X.height() * X.maps() * smX;
  const int offY = Y.width() * Y.height() * Y.maps() * smY;
  
  cl_int err =
    clblasSgemv(is_row_major ? clblasRowMajor : clblasColumnMajor,
      transpose_A ? clblasTrans : clblasNoTrans,
      M, N, alpha,
      (cl_mem)A.cl_data_ptr_, offA, ldA,
      (cl_mem)X.cl_data_ptr_, offX, incX, beta,
      (cl_mem)Y.cl_data_ptr_, offY, incY,
      1, &(CLHelper::queue), 0, NULL, &done_event);
  
  if(err!=CL_SUCCESS)
    FATAL("Call to clblasSgemv failed. Error: " << err);
  
  err = clWaitForEvents(1, &done_event);
  
  if(err!=CL_SUCCESS)
    FATAL("Call to clWaitForEvents failed. Error: " << err);
#else
#ifdef BUILD_BLAS
  INNERGEMV(is_row_major ? CblasRowMajor : CblasColMajor,
            transpose_A ? CblasTrans : CblasNoTrans,
            M, N, alpha, A.data_ptr_const(0,0,0,smA),
            ldA, X.data_ptr_const(0,0,0,smX), incX, beta, Y.data_ptr(0,0,0,smY), incY);
#else
  FATAL("No reference GEMV at this time!");
#endif // BUILD_BLAS
#endif // BUILD_CLBLAS
  Y.hint_ignore_content_ = false;
}


void TensorMath::IM2COL(const Tensor& source, const int source_width, const int source_height, const int maps, const int samples, const int kernel_width, const int kernel_height, const int stride_width, const int stride_height, const int pad_width, const int pad_height, Tensor& target)
{
#ifdef BUILD_OPENCL
  ((Tensor&)source).MoveToCPU();
  target.MoveToCPU(true);
#endif
  
  const int target_width = (2 * pad_width + source_width - kernel_width) / stride_width + 1;
  const int target_height = (2 * pad_height + source_height - kernel_height) / stride_height + 1;
  const int target_maps = kernel_width * kernel_height * maps;
  
  const int target_size = samples * target_width * target_height * target_maps;
  const int actual_target_size = target.samples() * target.width()* target.height() * target.maps();
  
  if(target_size != actual_target_size)
    FATAL("Target size wrong!");
  
  #pragma omp parallel for default(shared)
  for(int sample = 0; sample < samples; sample++) {
    const datum* source_ptr = source.data_ptr_const(0, 0, 0, sample);
    datum* target_ptr = target.data_ptr(0, 0, 0, sample);
    for(int target_map = 0; target_map < target_maps; target_map++) {
      int kx = target_map % kernel_width;
      int ky = (target_map / kernel_width) % kernel_height;
      int imap = target_map / (kernel_width * kernel_height);
      for(int oy = 0; oy < target_height; oy++) {
        int iy = oy * stride_height - pad_height + ky;
        if(iy >= 0 && iy < source_height) {
          for(int ox = 0; ox < target_width; ox++) {
            int ix = ox * stride_width - pad_width + kx;
            if(ix >= 0 && iy < source_width) {
              target_ptr[(target_map * target_height + oy) * target_width + ox] =
                source_ptr[(imap * source_height + iy) * source_width + ix];
            } else {
              target_ptr[(target_map * target_height + oy) * target_width + ox] = 0;
            }
          }
        } else {
          // Zero out
          for(int ox = 0; ox < target_width; ox++) {
              target_ptr[(target_map * target_height + oy) * target_width + ox] = 0;
          } 
        }
      }
    }
  }
  
  target.hint_ignore_content_ = false;
}

void TensorMath::COL2IM(Tensor& source, const int source_width, const int source_height, const int maps, const int samples, const int kernel_width, const int kernel_height, const int stride_width, const int stride_height, const int pad_width, const int pad_height, const Tensor& target)
{
#ifdef BUILD_OPENCL
  ((Tensor&)target).MoveToCPU();
  source.MoveToCPU(true);
#endif
  SETSAMPLE(source, -1, 0.0);
  
  const int target_width = (2 * pad_width + source_width - kernel_width) / stride_width + 1;
  const int target_height = (2 * pad_height + source_height - kernel_height) / stride_height + 1;
  const int target_maps = kernel_width * kernel_height * maps;
  
  const int target_size = samples * target_width * target_height * target_maps;
  const int actual_target_size = target.samples() * target.width()* target.height() * target.maps();
  
  if(target_size != actual_target_size)
    FATAL("Target size wrong!");
  
  #pragma omp parallel for default(shared)
  for(int sample = 0; sample < samples; sample++) {
    datum* source_ptr = source.data_ptr(0, 0, 0, sample);
    const datum* target_ptr = target.data_ptr_const(0, 0, 0, sample);
    for(int target_map = 0; target_map < target_maps; target_map++) {
      int kx = target_map % kernel_width;
      int ky = (target_map / kernel_width) % kernel_height;
      int imap = target_map / (kernel_width * kernel_height);
      for(int oy = 0; oy < target_height; oy++) {
        int iy = oy * stride_height - pad_height + ky;
        if(iy >= 0 && iy < source_height) {
          for(int ox = 0; ox < target_width; ox++) {
            int ix = ox * stride_width - pad_width + kx;
            if(ix >= 0 && iy < source_width) {
              source_ptr[(imap * source_height + iy) * source_width + ix] +=
                target_ptr[(target_map * target_height + oy) * target_width + ox];
            } 
          }
        }
      }
    }
  }
  
  source.hint_ignore_content_ = false;
}

void TensorMath::SETSAMPLE(Tensor& A, const int smA, const datum value)
{
#ifdef BUILD_OPENCL
  if(A.cl_gpu_) {
    cl_uint error = 0;
    cl_uint offset = smA == -1 ? 0 : smA * (A.width() * A.height() * A.maps());

    error |= clSetKernelArg (CLHelper::k_setValue, 0, sizeof (cl_mem), &(A.cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_setValue, 1, sizeof (datum), &value);
    error |= clSetKernelArg (CLHelper::k_setValue, 2, sizeof (cl_uint), &offset);

    if (error != CL_SUCCESS) {
      FATAL("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = {smA == -1 ? A.elements() : A.width() * A.height() * A.samples()};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_setValue, 1, NULL,
        global_work_size, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
      FATAL("Error enqueueing kernel: " << (signed int) error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (CLHelper::queue);
    if (error != CL_SUCCESS) {
      FATAL("Error finishing command queue: " << (signed int) error);
    }
#endif
      
  } else {
#endif
    datum* start = smA == -1 ? A.data_ptr() : A.data_ptr(0, 0, 0, smA);
    datum* end = smA == -1 ? A.data_ptr(0, 0, 0, A.samples()) : A.data_ptr(0, 0, 0, smA + 1);
    for(datum* ptr = start; ptr < end; ptr++)
      *ptr = value;
#ifdef BUILD_OPENCL
  }
#endif

  A.hint_ignore_content_ = false;
}

  
}