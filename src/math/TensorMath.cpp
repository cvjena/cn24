/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "MKLHelper.h"
#include "CLHelper.h"

#include <cstring>

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
#else
  
#ifdef BUILD_OPENCL
  ((Tensor&)A).MoveToCPU();
  ((Tensor&)B).MoveToCPU();
  C.MoveToCPU(C.hint_ignore_content_ && beta == 0.0);
#endif 
  
#ifdef BUILD_BLAS
  INNERGEMM(is_row_major ? CblasRowMajor : CblasColMajor,
    transpose_A ? CblasTrans : CblasNoTrans,
    transpose_B ? CblasTrans : CblasNoTrans,
    M, N, K,
    alpha, A.data_ptr_const(0,0,0,smA), ldA,
    B.data_ptr_const(0,0,0,smB), ldB,
    beta, C.data_ptr(0,0,0,smC), ldC);
#else
  if(!is_row_major)
    FATAL("Reference GEMM does not support column-major matrices!");
  
  const datum* a_ptr = A.data_ptr_const(0, 0, 0, smA);
  const datum* b_ptr = B.data_ptr_const(0, 0, 0, smB);
  datum* c_ptr = C.data_ptr(0, 0, 0, smC);
  
  #pragma omp parallel for default(shared)
  for(int i = 0; i < M; i++) {
    for(int j = 0; j < N; j++) {
      datum sum = 0.0;
      for(int k = 0; k < K; k++) {
        const datum a_value = transpose_A ?
          a_ptr[k * ldA + i]
        :
          a_ptr[i * ldA + k];
        
        const datum b_value = transpose_B ?
          b_ptr[j * ldB + k]
        :
          b_ptr[k * ldB + j];
          
        sum += a_value * b_value;
      }
      if(beta == 0.0)
        c_ptr[ldC * i + j] = alpha * sum;
      else
        c_ptr[ldC * i + j] = beta * c_ptr[ldC * i + j] + alpha * sum;
    }
  }
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
#else
#ifdef BUILD_OPENCL
  ((Tensor&)A).MoveToCPU();
  ((Tensor&)X).MoveToCPU();
  Y.MoveToCPU(Y.hint_ignore_content_ && beta == 0.0);
#endif
  
#ifdef BUILD_BLAS
  INNERGEMV(is_row_major ? CblasRowMajor : CblasColMajor,
            transpose_A ? CblasTrans : CblasNoTrans,
            M, N, alpha, A.data_ptr_const(0,0,0,smA),
            ldA, X.data_ptr_const(0,0,0,smX), incX, beta, Y.data_ptr(0,0,0,smY), incY);
#else
  if(!is_row_major)
    FATAL("Reference GEMV does not support column-major matrices!");
  
  // ...
  const datum* a_ptr = A.data_ptr_const(0, 0, 0, smA);
  const datum* x_ptr = X.data_ptr_const(0, 0, 0, smX);
  datum* y_ptr = Y.data_ptr(0, 0, 0, smY);
  
  #pragma omp parallel for default(shared)
  for(int i = 0; i < M; i++) {
    datum sum = 0.0;
    for(int j = 0; j < N; j++) {
      const datum a_value = transpose_A ?
        a_ptr[j * ldA + i]
      :
        a_ptr[i * ldA + j];
      
      const datum x_value = x_ptr[j * incX];
      sum += x_value * a_value;
    }
    if(beta == 0.0)
      y_ptr[i * incY] = alpha * sum;
    else 
      y_ptr[i * incY] = beta * y_ptr[i * incY] + alpha * sum;
  }
  
#endif // BUILD_BLAS
#endif // BUILD_CLBLAS
  Y.hint_ignore_content_ = false;
}


void TensorMath::IM2COL(const Tensor& source, const int source_width, const int source_height, const int maps, const int samples, const int kernel_width, const int kernel_height, const int stride_width, const int stride_height, const int pad_width, const int pad_height, Tensor& target)
{
#ifdef BUILD_OPENCL
  if(source.cl_gpu_ || target.cl_gpu_) {
    ((Tensor&)source).MoveToGPU();
    target.MoveToGPU(true);

    cl_uint error = 0;
    const int target_width = (2 * pad_width + source_width - kernel_width) / stride_width + 1;
    const int target_height = (2 * pad_height + source_height - kernel_height) / stride_height + 1;
    const int target_maps = kernel_width * kernel_height * maps;
    
    error |= clSetKernelArg (CLHelper::k_im2col, 0, sizeof (cl_mem), &(((Tensor&)source).cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_im2col, 1, sizeof (cl_mem), &(target.cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_im2col, 2, sizeof (cl_int), &source_width);
    error |= clSetKernelArg (CLHelper::k_im2col, 3, sizeof (cl_int), &source_height);
    error |= clSetKernelArg (CLHelper::k_im2col, 4, sizeof (cl_int), &maps);
    error |= clSetKernelArg (CLHelper::k_im2col, 5, sizeof (cl_int), &samples);
    error |= clSetKernelArg (CLHelper::k_im2col, 6, sizeof (cl_int), &target_width);
    error |= clSetKernelArg (CLHelper::k_im2col, 7, sizeof (cl_int), &target_height);
    error |= clSetKernelArg (CLHelper::k_im2col, 8, sizeof (cl_int), &target_maps);
    error |= clSetKernelArg (CLHelper::k_im2col, 9, sizeof (cl_int), &kernel_width);
    error |= clSetKernelArg (CLHelper::k_im2col, 10, sizeof (cl_int), &kernel_height);
    error |= clSetKernelArg (CLHelper::k_im2col, 11, sizeof (cl_int), &stride_width);
    error |= clSetKernelArg (CLHelper::k_im2col, 12, sizeof (cl_int), &stride_height);
    error |= clSetKernelArg (CLHelper::k_im2col, 13, sizeof (cl_int), &pad_width);
    error |= clSetKernelArg (CLHelper::k_im2col, 14, sizeof (cl_int), &pad_height);

    if (error != CL_SUCCESS) {
      FATAL("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = {(size_t)(target_width * target_height), (size_t)target_maps, (size_t)samples};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_im2col, 3, NULL,
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
    ((Tensor&)source).MoveToCPU();
    target.MoveToCPU(true);
#endif
  
    const int target_width = (2 * pad_width + source_width - kernel_width) / stride_width + 1;
    const int target_height = (2 * pad_height + source_height - kernel_height) / stride_height + 1;
    const int target_maps = kernel_width * kernel_height * maps;
    
    const int target_size = samples * target_width * target_height * target_maps;
    const int actual_target_size = target.samples() * target.width() * target.height() * target.maps();
    
    if(target_size != actual_target_size)
      FATAL("Target size wrong!");
    
    
    #pragma omp parallel for default(shared)
    for(int sample = 0; sample < samples; sample++) {
      const datum* source_ptr = source.data_ptr_const(0, 0, 0, sample);
      for(int target_map = 0; target_map < target_maps; target_map++) {
        datum* target_ptr = target.data_ptr(0, 0, 0, target_map); 
        int kx = target_map % kernel_width;
        int ky = (target_map / kernel_width) % kernel_height;
        int imap = target_map / (kernel_width * kernel_height);
        for(int oy = 0; oy < target_height; oy++) {
          int iy = oy * stride_height - pad_height + ky;
          if(iy >= 0 && iy < source_height) {
            for(int ox = 0; ox < target_width; ox++) {
              int ix = ox * stride_width - pad_width + kx;
              if(ix >= 0 && ix < source_width) {
                target_ptr[(sample * target_height + oy) * target_width + ox] =
                  source_ptr[(imap * source_height + iy) * source_width + ix];
              } else {
                target_ptr[(sample * target_height + oy) * target_width + ox] = 0;
              }
            }
          } else {
            // Zero out
            for(int ox = 0; ox < target_width; ox++) {
                target_ptr[(sample * target_height + oy) * target_width + ox] = 0;
            } 
          }
        }
      }
    }
    
#ifdef BUILD_OPENCL
  }
#endif
  
  target.hint_ignore_content_ = false;
}

void TensorMath::COL2IM(Tensor& source, const int source_width, const int source_height, const int maps, const int samples, const int kernel_width, const int kernel_height, const int stride_width, const int stride_height, const int pad_width, const int pad_height, const Tensor& target)
{
#ifdef BUILD_OPENCL
  if(source.cl_gpu_ || target.cl_gpu_) {
    ((Tensor&)target).MoveToGPU();
    source.MoveToGPU(true);
    
    cl_uint error = 0;
    const int target_width = (2 * pad_width + source_width - kernel_width) / stride_width + 1;
    const int target_height = (2 * pad_height + source_height - kernel_height) / stride_height + 1;
    const int target_maps = kernel_width * kernel_height * maps;
    
    error |= clSetKernelArg (CLHelper::k_col2im, 0, sizeof (cl_mem), &(((Tensor&)source).cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_col2im, 1, sizeof (cl_mem), &(target.cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_col2im, 2, sizeof (cl_int), &source_width);
    error |= clSetKernelArg (CLHelper::k_col2im, 3, sizeof (cl_int), &source_height);
    error |= clSetKernelArg (CLHelper::k_col2im, 4, sizeof (cl_int), &maps);
    error |= clSetKernelArg (CLHelper::k_col2im, 5, sizeof (cl_int), &samples);
    error |= clSetKernelArg (CLHelper::k_col2im, 6, sizeof (cl_int), &target_width);
    error |= clSetKernelArg (CLHelper::k_col2im, 7, sizeof (cl_int), &target_height);
    error |= clSetKernelArg (CLHelper::k_col2im, 8, sizeof (cl_int), &target_maps);
    error |= clSetKernelArg (CLHelper::k_col2im, 9, sizeof (cl_int), &kernel_width);
    error |= clSetKernelArg (CLHelper::k_col2im, 10, sizeof (cl_int), &kernel_height);
    error |= clSetKernelArg (CLHelper::k_col2im, 11, sizeof (cl_int), &stride_width);
    error |= clSetKernelArg (CLHelper::k_col2im, 12, sizeof (cl_int), &stride_height);
    error |= clSetKernelArg (CLHelper::k_col2im, 13, sizeof (cl_int), &pad_width);
    error |= clSetKernelArg (CLHelper::k_col2im, 14, sizeof (cl_int), &pad_height);

    if (error != CL_SUCCESS) {
      FATAL("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = {(size_t)(source_width * source_height), (size_t)maps, (size_t)samples};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_col2im, 3, NULL,
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
    
    for(int sample = 0; sample < samples; sample++) {
      datum* source_ptr = source.data_ptr(0, 0, 0, sample);
      for(int target_map = 0; target_map < target_maps; target_map++) {
        const datum* target_ptr = target.data_ptr_const(0, 0, 0, target_map);
        int kx = target_map % kernel_width;
        int ky = (target_map / kernel_width) % kernel_height;
        int imap = target_map / (kernel_width * kernel_height);
        for(int oy = 0; oy < target_height; oy++) {
          int iy = oy * stride_height - pad_height + ky;
          if(iy >= 0 && iy < source_height) {
            for(int ox = 0; ox < target_width; ox++) {
              int ix = ox * stride_width - pad_width + kx;
              if(ix >= 0 && ix < source_width) {
                source_ptr[(imap * source_height + iy) * source_width + ix] +=
                  target_ptr[(sample * target_height + oy) * target_width + ox];
              } 
            }
          }
        }
      }
    }
  
#ifdef BUILD_OPENCL
  }
#endif
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

    size_t global_work_size[] = {(size_t)(smA == -1 ? A.elements() : A.width() * A.height() * A.samples())};

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

void TensorMath::SMS(const Tensor& source, Tensor& target)
{
#ifdef BUILD_OPENCL
  if(source.cl_gpu_ || target.cl_gpu_) {
    ((Tensor&)source).MoveToGPU();
    target.MoveToGPU(true);
    const int width = target.width();
    const int height = target.height();
    const int maps = target.maps();
    const int samples = target.samples();

    cl_uint error = 0;

    error |= clSetKernelArg (CLHelper::k_sms, 0, sizeof (cl_mem), &(((Tensor&)source).cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_sms, 1, sizeof (cl_mem), &(target.cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_sms, 2, sizeof (cl_uint), &width);
    error |= clSetKernelArg (CLHelper::k_sms, 3, sizeof (cl_uint), &height);
    error |= clSetKernelArg (CLHelper::k_sms, 4, sizeof (cl_uint), &maps);
    error |= clSetKernelArg (CLHelper::k_sms, 5, sizeof (cl_uint), &samples);

    if (error != CL_SUCCESS) {
      FATAL("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = {(size_t)target.elements()};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_sms, 1, NULL,
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
    const int width = target.width();
    const int height = target.height();
    const int maps = target.maps();
    const int samples = target.samples();
    for(int sample = 0; sample < samples; sample++) {
      for(int map = 0; map < maps; map++) {
        const datum* src = source.data_ptr_const(0, 0, sample, map);
        datum* tgt = target.data_ptr(0, 0, map, sample);
        std::memcpy(tgt, src, sizeof(datum) * width * height);
      }
    }
  
#ifdef BUILD_OPENCL
  }
#endif
  
  target.hint_ignore_content_ = false;
}

void TensorMath::DOWN(const Tensor& source, Tensor& target, const int region_width, const int region_height, const datum target_factor)
{
#ifdef BUILD_OPENCL
  if(source.cl_gpu_ || target.cl_gpu_) {
    ((Tensor&)source).MoveToGPU();
    target.MoveToGPU(true);
    const int target_width = target.width();
    const int target_height = target.height();
    const int source_width = source.width();
    const int source_height = source.height();
    cl_int error = 0;

    error |= clSetKernelArg (CLHelper::k_down, 0, sizeof (cl_mem), &(((Tensor&)source).cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_down, 1, sizeof (cl_mem), &(target.cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_down, 2, sizeof (cl_uint), &target_width);
    error |= clSetKernelArg (CLHelper::k_down, 3, sizeof (cl_uint), &target_height);
    error |= clSetKernelArg (CLHelper::k_down, 4, sizeof (cl_uint), &source_width);
    error |= clSetKernelArg (CLHelper::k_down, 5, sizeof (cl_uint), &source_height);
    error |= clSetKernelArg (CLHelper::k_down, 6, sizeof (cl_uint), &region_width);
    error |= clSetKernelArg (CLHelper::k_down, 7, sizeof (cl_uint), &region_height);
    error |= clSetKernelArg (CLHelper::k_down, 8, sizeof (cl_float), &target_factor);

    if (error != CL_SUCCESS) {
      FATAL("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = {(size_t)target.width(), (size_t)target.height(), (size_t)(target.maps() * target.samples())};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_down, 3, NULL,
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
    const int target_width = (int)target.width();
    const int target_height = (int)target.height();
    const int maps = (int)target.maps();
    const int samples = (int)target.samples();
    for(int sample = 0; sample < samples; sample++) {
      for(int map = 0; map < maps; map++) {
        for(int target_y = 0; target_y < target_height; target_y++) {
          const int source_y = region_height * target_y;
          for(int target_x = 0; target_x < target_width; target_x++) {
            const int source_x = region_width * target_x;
            datum sum = 0;
            for(int ry = 0; ry < region_height; ry++) {
              for(int rx = 0; rx < region_width; rx++) {
                const datum* src = source.data_ptr_const((const size_t)(source_x + rx), (const size_t)(source_y + ry), (const size_t)map, (const size_t)sample);
                sum += *src;
              }
            }
            datum* tgt = target.data_ptr((const size_t)target_x, (const size_t)target_y, (const size_t)map, (const size_t)sample);
            *tgt = sum * target_factor;
          }
        }
      }
    }
    
#ifdef BUILD_OPENCL
  }
#endif

  target.hint_ignore_content_ = false;
}

void TensorMath::UP(const Tensor& source, Tensor& target, const int region_width, const int region_height, const datum target_factor)
{
#ifdef BUILD_OPENCL
  if(source.cl_gpu_ || target.cl_gpu_) {
    ((Tensor&)source).MoveToGPU();
    target.MoveToGPU(true);
    const int target_width = (int)target.width();
    const int target_height = (int)target.height();
    const int source_width = (int)source.width();
    const int source_height = (int)source.height();
    cl_int error = 0;

    error |= clSetKernelArg (CLHelper::k_up, 0, sizeof (cl_mem), &(((Tensor&)source).cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_up, 1, sizeof (cl_mem), &(target.cl_data_ptr_));
    error |= clSetKernelArg (CLHelper::k_up, 2, sizeof (cl_uint), &target_width);
    error |= clSetKernelArg (CLHelper::k_up, 3, sizeof (cl_uint), &target_height);
    error |= clSetKernelArg (CLHelper::k_up, 4, sizeof (cl_uint), &source_width);
    error |= clSetKernelArg (CLHelper::k_up, 5, sizeof (cl_uint), &source_height);
    error |= clSetKernelArg (CLHelper::k_up, 6, sizeof (cl_uint), &region_width);
    error |= clSetKernelArg (CLHelper::k_up, 7, sizeof (cl_uint), &region_height);
    error |= clSetKernelArg (CLHelper::k_up, 8, sizeof (cl_float), &target_factor);

    if (error != CL_SUCCESS) {
      FATAL("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = {(size_t)target.width(), (size_t)target.height(), (size_t)(target.maps() * target.samples())};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_up, 3, NULL,
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
    const int width = (int)source.width();
    const int height = (int)source.height();
    const int maps = (int)source.maps();
    const int samples = (int)source.samples();
    for(int sample = 0; sample < samples; sample++) {
      for(int map = 0; map < maps; map++) {
        for(int y = 0; y < height; y++) {
          const int iy = region_height * y;
          for(int x = 0; x < width; x++) {
            const int ix = region_width * x;
            const datum* src = source.data_ptr_const((const size_t)x, (const size_t)y, (const size_t)map, (const size_t)sample);
            datum sum = *src;
            for(int ry = 0; ry < region_height; ry++) {
              for(int rx = 0; rx < region_width; rx++) {
                datum* tgt = target.data_ptr((const size_t)(ix + rx), (const size_t)(iy + ry), (const size_t)map, (const size_t)sample);
                *tgt = sum * target_factor;
              }
            }
          }
        }
      }
    }
#ifdef BUILD_OPENCL
  }
#endif

  target.hint_ignore_content_ = false;
}

void TensorMath::ADD(const Tensor& source_a, const Tensor& source_b, Tensor& target)
{
#ifdef BUILD_OPENCL
  ((Tensor&)source_a).MoveToCPU();
  ((Tensor&)source_b).MoveToCPU();
  target.MoveToCPU(true);
#endif
  if((source_a.samples() != source_b.samples())
    || (source_b.samples() != target.samples())
    || (source_a.elements() != source_b.elements())
    || (source_b.elements() != target.elements())) {
    FATAL("Dimensions don't match!");
  }
  
  #pragma omp parallel for default(shared)
  for(unsigned int element = 0; element < source_a.elements(); element++) {
    const datum* source_a_ptr = &(source_a.data_ptr_const()[element]);
    const datum* source_b_ptr = &(source_b.data_ptr_const()[element]);
    datum* target_ptr = &(target.data_ptr()[element]);
    
    *target_ptr = *source_a_ptr + *source_b_ptr;
  }
  
  target.hint_ignore_content_ = false;
}


}
