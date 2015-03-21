/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cstring>
#include <algorithm>

#ifdef BUILD_OPENCL
#define BUILD_OPENCL_CONV
#endif

#ifdef BUILD_OPENCL_CONV
#include <iomanip>
#endif

#include "Config.h"
#include "Log.h"
#include "Net.h"
#include "CLHelper.h"
#include "MKLHelper.h"

#include "ConvolutionLayer.h"

namespace Conv {

ConvolutionLayer::ConvolutionLayer (const unsigned int kwidth,
                                    const unsigned int kheight,
                                    const unsigned int output_maps,
                                    const int seed, const datum dropout_fraction) :
  output_maps_ (output_maps), kernel_width_ (kwidth), kernel_height_ (kheight),
  rand_ (seed), dropout_fraction_(dropout_fraction) {
  // Validate kernel dimensions. These are very important because the
  // FeedForward and BackPropagate implementations rely on some assumptions.

  // The kernels must not be of zero size in any dimension.
  if (kernel_width_ == 0 || kernel_height_ == 0) {
    FATAL ("Kernels cannot have zero dimensions");
  }

  // Using a zero seed on more than one layer may introduce symmetries and
  // influence the gain of a network in a negative way
  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }

  LOGDEBUG << "Instance created. " << output_maps_ << " output maps with " <<
           kernel_width_ << "x" << kernel_height_ << " kernels.";
  LOGDEBUG << "Dropout fraction: " << dropout_fraction_;
}

bool ConvolutionLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs) {
  // This is a simple layer, only one input
  if (inputs.size() != 1) {
    LOGERROR << "Only one input supported!";
    return false;
  }

  // Save input node pointer
  CombinedTensor* input = inputs[0];

  // Check if input node pointer is null
  if (input == nullptr) {
    LOGERROR << "Null pointer input node!";
    return false;
  }

  // Validate dimensions
  // The input maps have to be larger or at least as large as the kernel,
  // because we only do 'valid' convolutions.
  // For MNIST recognition, LeCun says that 'same' convolutions perform better
  // as a first layer. He adds a border around the training and test sets to
  // achieve the same thing without changing the code.
  if (input->data.height() < kernel_height_ || input->data.width() < kernel_width_) {
    LOGERROR << "Unsupported input dimensions " << input->data;
    return false;
  }

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      input->data.width() - (kernel_width_ - 1), input->data.height() - (kernel_height_ - 1),
      output_maps_);

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool ConvolutionLayer::Connect (const CombinedTensor* input,
                                CombinedTensor* output) {
  bool valid =
    input->data.width() >= kernel_width_ && input->data.height() >=  kernel_height_ &&
    output->data.width() == input->data.width() - (kernel_width_ - 1) &&
    output->data.height() == input->data.height() - (kernel_height_ - 1);

  if (!valid) {
    return false;
  }

  // Save parameters
  input_maps_ = input->data.maps();
  input_width_ = input->data.width();
  input_height_ = input->data.height();
  output_width_ = output->data.width();
  output_height_ = output->data.height();

  /*LOGDEBUG << "Local learning rate setting was " << local_lr_;
  local_lr_ /= (datum)(output_width_ * output_height_);*/
  LOGDEBUG << "Local learning rate is now " << local_lr_;
#ifdef BUILD_BLAS
  // Create im2col output buffer
  im2col_ff_buffer.Resize (input->data.samples(), kernel_width_ * kernel_height_, input_maps_,
                           output_width_ * output_height_);

  // Create FeedForward output buffer
  ff_output_buffer.Resize (output_maps_, output_width_, output_height_,
                           input->data.samples());

  // Create backpropagation input buffer
  bp_deltay_buffer.Resize (output_maps_, output_width_, output_height_,
                           input->data.samples());

  bp_deltax_buffer.Resize (input->data.samples(), kernel_width_ * kernel_height_, input_maps_,
                           output_width_ * output_height_);

  // This is faster than adding manually...
  ones_.Resize (1, output_width_ * output_height_ * input->data.samples());

  for (unsigned int i = 0; i < ones_.elements(); i++) {
    ones_[i] = 1;
  }

#endif

#ifdef BUILD_OPENCL_CONV
  // Create folding buffers for OpenCL
  delta_buffer_.Resize (input->data.samples(), kernel_width_, kernel_height_, input_maps_ * output_maps_);
  bias_buffer_.Resize (input->data.samples(), output_maps_);
#endif

  // Create kernels
  weights_ = new CombinedTensor (output_maps_, kernel_width_, kernel_height_, input_maps_);
  bias_ = new CombinedTensor (1, output_maps_);

  // Initialize weights to zero so the net won't work if Net::InitializeWeights
  // is not called. Random memory junk may work but is certainly not optimal.
  bias_->data.Clear();
  weights_->data.Clear();
  
  // Initialize the dropout mask tensor
  dropout_mask_.Resize(input->data.samples(),output_maps_);

  // Tell the net about our parameters
  parameters_.push_back (weights_);
  parameters_.push_back (bias_);

  return true;
}

void ConvolutionLayer::FeedForward() {
  const datum p = net_->IsTesting() ? 0.0 : dropout_fraction_;
  const datum w = net_->IsTesting() ? (1.0 - dropout_fraction_) : 1.0;
  
#ifdef BUILD_OPENCL_CONV
  dropout_mask_.MoveToCPU();
#endif
  if(p == 0.0) {
    dropout_mask_.Clear(1.0);
  } else {
    std::uniform_real_distribution<datum> dist(0.0, 1.0);
    for(unsigned int e = 0; e < input_->data.samples() * output_maps_; e++) {
      dropout_mask_[e] = dist(rand_) < p ? 0.0 : 1.0;
    }
  }
#ifdef BUILD_OPENCL_CONV
  dropout_mask_.MoveToGPU();
#endif
  
  // Because we add every input map
  output_->data.Clear();

#ifdef BUILD_OPENCL_CONV
  cl_uint error = 0;
  input_->data.MoveToGPU();
  weights_->data.MoveToGPU();
  bias_->data.MoveToGPU();
  output_->data.MoveToGPU (true);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 1, sizeof (cl_mem), &weights_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 2, sizeof (cl_mem), &bias_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 3, sizeof (cl_mem), &output_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 4, sizeof (cl_mem), &dropout_mask_.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 5, sizeof (unsigned int), &input_width_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 6, sizeof (unsigned int), &input_height_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 7, sizeof (unsigned int), &input_maps_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 8, sizeof (unsigned int), &kernel_width_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 9, sizeof (unsigned int), &kernel_height_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 10, sizeof (unsigned int), &output_width_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 11, sizeof (unsigned int), &output_height_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 12, sizeof (unsigned int), &output_maps_);
  error |= clSetKernelArg (CLHelper::k_biasedConvolution, 13, sizeof (datum), &w);

  if (error != CL_SUCCESS) {
    FATAL ("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = { output_width_, output_height_, output_maps_* input_->data.samples() };

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_biasedConvolution, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);

  if (error != CL_SUCCESS) {
    FATAL ("Error enqueueing kernel: " << (signed int) error);
  }

#ifdef BRUTAL_FINISH
  error = clFinish (CLHelper::queue);

  if (error != CL_SUCCESS) {
    FATAL ("Error finishing command queue: " << (signed int) error);
  }

#endif

#else // No OpenCL

  // Very simple dropout FF implementation
  // This could be optimized a _lot_
  unsigned int sk_id = 0;
  for(unsigned int s = 0; s < input_->data.samples(); s++) {
    for(unsigned int m = 0; m < output_maps_; m++) {
      datum* target_map = output_->data.data_ptr(0,0,m,s);
      for(unsigned int e = 0; e < output_width_ * output_height_; e++) {
        if(dropout_mask_[sk_id] == 0.0)
          target_map[e] = 0.0;
      }
    }
    sk_id++;
  }

#ifdef BUILD_BLAS
  im2colff();

  const datum* W = weights_->data.data_ptr_const();
  const datum* X = im2col_ff_buffer.data_ptr_const();
  const datum* b = bias_->data.data_ptr_const();
  datum* Y = ff_output_buffer.data_ptr();

  // Convolve
  GEMM (CblasRowMajor, CblasNoTrans, CblasTrans, output_maps_,
        output_width_ * output_height_ * input_->data.samples(),
        kernel_width_ * kernel_height_ * input_maps_,
        w , W, kernel_width_ * kernel_height_ * input_maps_,
        X, kernel_width_ * kernel_height_ * input_maps_, 0.0,
        Y, output_width_ * output_height_ * input_->data.samples());

  // Add bias
  GEMM (CblasRowMajor, CblasNoTrans, CblasNoTrans, output_maps_,
        output_width_ * output_height_ * input_->data.samples(), 1, 1.0, b, 1,
        ones_.data_ptr_const(), output_width_ * output_height_ * input_->data.samples(),
        1.0, Y, output_width_ * output_height_ * input_->data.samples());


  col2imff();
#else
  // i * k = o
  // The kernels are always "flipped", so this is an actual convolution
  #pragma omp parallel for default(shared)

  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int omap = 0; omap < output_maps_; omap++) {
      datum bias = bias_->data (omap);

      for (unsigned int oy = 0; oy < output_height_; oy++) {
        for (unsigned int ox = 0; ox < output_width_; ox++) {
          datum* oval =
            output_->data.data_ptr (ox, oy, omap, sample);

          for (unsigned int ky = 0; ky < kernel_height_; ky++) {
            unsigned int iy = oy + ky;

            for (unsigned int kx = 0; kx < kernel_width_; kx++) {
              unsigned int ix = ox + kx;

              for (unsigned int imap = 0; imap < input_maps_; imap++) {
                const datum weight =
                  *weights_->data.data_ptr_const (kx, ky, imap, omap);
                const datum ival =
                  *input_->data.data_ptr_const (ix, iy, imap, sample);

                *oval += w * weight * ival;
              }
            }
          }

          *oval += bias;
        }
      }
    }
  }

#endif // else BUILD_BLAS
#endif // else BUILD_OPENCL
}

void ConvolutionLayer::BackPropagate() {

  static datum one = 1.0;

#ifndef BUILD_OPENCL_CONV
#ifndef BUILD_BLAS
  weights_->delta.Clear();
  input_->delta.Clear();
#endif
#endif

#ifdef BUILD_OPENCL_CONV
  output_->delta.MoveToCPU();
#endif
  // Very simple dropout backprop implementation
  // This could be optimized a _lot_
  unsigned int sk_id = 0;
  for(unsigned int s = 0; s < input_->data.samples(); s++) {
    for(unsigned int m = 0; m < output_maps_; m++) {
      datum* target_map = output_->delta.data_ptr(0,0,m,s);
      if(dropout_mask_[sk_id] == 0.0)
        for(unsigned int e = 0; e < output_width_ * output_height_; e++) {
          target_map[e] = 0.0;
        }
    }
    sk_id++;
  }

  /*
   * 1. Backpropagation
   */
#ifdef BUILD_OPENCL_CONV

  if (backprop_enabled_) {
    // dX = dY * W'
    cl_uint error = 0;

    output_->delta.MoveToGPU();
    input_->delta.MoveToGPU (true);
    weights_->data.MoveToGPU();

    error |= clSetKernelArg (CLHelper::k_fullConvolution, 0, sizeof (cl_mem), &output_->delta.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 1, sizeof (cl_mem), &weights_->data.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 2, sizeof (cl_mem), &input_->delta.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 3, sizeof (unsigned int), &input_width_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 4, sizeof (unsigned int), &input_height_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 5, sizeof (unsigned int), &input_maps_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 6, sizeof (unsigned int), &kernel_width_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 7, sizeof (unsigned int), &kernel_height_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 8, sizeof (unsigned int), &output_width_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 9, sizeof (unsigned int), &output_height_);
    error |= clSetKernelArg (CLHelper::k_fullConvolution, 10, sizeof (unsigned int), &output_maps_);

    if (error != CL_SUCCESS) {
      FATAL ("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = { input_width_, input_height_, input_maps_* input_->data.samples() };

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_fullConvolution, 3, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);

    if (error != CL_SUCCESS) {
      FATAL ("Error enqueueing kernel: " << (signed int) error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (CLHelper::queue);

    if (error != CL_SUCCESS) {
      FATAL ("Error finishing command queue: " << (signed int) error);
    }

#endif

  }

#else
#ifdef BUILD_BLAS
  im2colbp();

  const datum* W = weights_->data.data_ptr_const();
  datum* dX = bp_deltax_buffer.data_ptr();
  const datum* dY = bp_deltay_buffer.data_ptr_const();
  const datum* X = im2col_ff_buffer.data_ptr_const();
  datum* dW = weights_->delta.data_ptr();

  if (backprop_enabled_) {
    GEMM (CblasRowMajor, CblasTrans, CblasNoTrans,
          output_width_ * output_height_ * input_->data.samples(),
          kernel_width_ * kernel_height_ * input_maps_, output_maps_, 1.0,
          dY, output_width_ * output_height_ * input_->data.samples(),
          W, kernel_width_ * kernel_height_ * input_maps_, 0.0,
          dX, kernel_width_ * kernel_height_ * input_maps_);


    input_->delta.Clear();
    col2imbp();
  }

#else

  const unsigned int inner_nh_x = kernel_width_ - 1;
  const unsigned int inner_nh_y = kernel_height_ - 1;

  // Backpropagate by "full"-convolving the output gradient with the
  // flipped filter.
  if (backprop_enabled_) {
    #pragma omp parallel for default(shared)

    for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
      for (unsigned int imap = 0; imap < input_maps_; imap++) {
        for (unsigned int diy = 0; diy < input_height_; diy++) {
          for (unsigned int dix = 0; dix < input_width_; dix++) {
            for (unsigned int ky = 0; ky < kernel_height_; ky++) {
              int doy = ky + diy - inner_nh_y;

              for (unsigned int kx = 0; kx < kernel_width_; kx++) {
                int dox = kx + dix - inner_nh_x;

                for (unsigned int omap = 0; omap < output_maps_; omap++) {
                  if (dox >= 0 && doy >= 0 && dox < (int) output_width_ &&
                      doy < (int) output_height_) {
                    const datum weight =
                      *weights_->data.data_ptr_const (kernel_width_ - (kx + 1),
                                                      kernel_height_ - (ky + 1), imap,
                                                      omap);
                    const datum doval =
                      *output_->delta.data_ptr_const (dox, doy, omap, sample);

                    datum* dival = input_->delta.data_ptr (dix, diy, imap, sample);

                    *dival += doval * weight;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

#endif // BUILD_BLAS
#endif // BUILD_OPENCL

  /*
   * 2. Weight gradient calculation
   */
#ifdef BUILD_OPENCL_CONV
  {
    cl_uint error = 0;
    input_->data.MoveToGPU();
    delta_buffer_.MoveToGPU (true);
    output_->delta.MoveToGPU();
    const unsigned int samples = input_->data.samples();
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 1, sizeof (cl_mem), &output_->delta.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 2, sizeof (cl_mem), &delta_buffer_.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 3, sizeof (unsigned int), &input_height_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 4, sizeof (unsigned int), &input_width_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 5, sizeof (unsigned int), &input_maps_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 6, sizeof (unsigned int), &output_width_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 7, sizeof (unsigned int), &output_height_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 8, sizeof (unsigned int), &output_maps_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 9, sizeof (unsigned int), &kernel_width_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 10, sizeof (unsigned int), &kernel_height_);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 11, sizeof (unsigned int), &samples);
    error |= clSetKernelArg (CLHelper::k_crossCorrelation, 12, sizeof (datum), &one);

    if (error != CL_SUCCESS) {
      FATAL ("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = { kernel_width_, kernel_height_, input_maps_* output_maps_ * samples};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_crossCorrelation, 3, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);

    if (error != CL_SUCCESS) {
      FATAL ("Error enqueueing kernel: " << (signed int) error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (CLHelper::queue);

    if (error != CL_SUCCESS) {
      FATAL ("Error finishing command queue: " << (signed int) error);
    }

#endif

  }

  {

    cl_uint error = 0;
    weights_->delta.MoveToGPU (true);
    delta_buffer_.MoveToGPU();

    const unsigned int samples = input_->data.samples();
    error |= clSetKernelArg (CLHelper::k_foldWeights, 0, sizeof (cl_mem), &delta_buffer_.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_foldWeights, 1, sizeof (cl_mem), &weights_->delta.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_foldWeights, 2, sizeof (unsigned int), &input_maps_);
    error |= clSetKernelArg (CLHelper::k_foldWeights, 3, sizeof (unsigned int), &output_maps_);
    error |= clSetKernelArg (CLHelper::k_foldWeights, 4, sizeof (unsigned int), &kernel_width_);
    error |= clSetKernelArg (CLHelper::k_foldWeights, 5, sizeof (unsigned int), &kernel_height_);
    error |= clSetKernelArg (CLHelper::k_foldWeights, 6, sizeof (unsigned int), &samples);

    if (error != CL_SUCCESS) {
      FATAL ("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = { kernel_width_, kernel_height_, input_maps_ * output_maps_};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_foldWeights, 3, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);

    if (error != CL_SUCCESS) {
      FATAL ("Error enqueueing kernel: " << (signed int) error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (CLHelper::queue);

    if (error != CL_SUCCESS) {
      FATAL ("Error finishing command queue: " << (signed int) error);
    }

#endif

  }

#else
#ifdef BUILD_BLAS
  GEMM (CblasRowMajor, CblasNoTrans, CblasNoTrans, output_maps_,
        kernel_width_ * kernel_height_ * input_maps_,
        output_width_ * output_height_ * input_->data.samples(),
        1.0, dY, output_width_ * output_height_ * input_->data.samples(),
        X, kernel_width_ * kernel_height_ * input_maps_,
        0.0, dW, kernel_width_ * kernel_height_ * input_maps_);
#else

  // Calculate gradients via x-correlation i * do = k
  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int imap = 0; imap < input_maps_; imap++) {
      for (unsigned int ky = 0; ky < kernel_height_; ky++) {
        for (unsigned int kx = 0; kx < kernel_width_; kx++) {
          for (unsigned int doy = 0; doy < output_height_; doy++) {
            unsigned int iy = ky + doy;

            for (unsigned int dox = 0; dox < output_width_; dox++) {
              unsigned int ix = kx + dox;

              for (unsigned int omap = 0; omap < output_maps_; omap++) {
                const datum doval =
                  *output_->delta.data_ptr_const (dox, doy, omap, sample);
                const datum ival =
                  *input_->data.data_ptr_const (ix, iy, imap, sample);
                // Don't flip here
                datum* kval =
                  weights_->delta.data_ptr (kx, ky, imap, omap);

                *kval += doval * ival;
              }
            }
          }
        }
      }
    }
  }

#endif // BUILD_BLAS
#endif // BUILD_OPENCL
  /*
  * 3. Bias gradient calculation
  */
#ifdef BUILD_OPENCL_CONV
  {
    cl_uint error = 0;
    bias_buffer_.MoveToGPU (true);
    output_->delta.MoveToGPU();
    const unsigned int samples = input_->data.samples();
    error |= clSetKernelArg (CLHelper::k_biasGradientPart1, 0, sizeof (cl_mem), &output_->delta.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_biasGradientPart1, 1, sizeof (cl_mem), &bias_buffer_.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_biasGradientPart1, 2, sizeof (unsigned int), &output_width_);
    error |= clSetKernelArg (CLHelper::k_biasGradientPart1, 3, sizeof (unsigned int), &output_height_);
    error |= clSetKernelArg (CLHelper::k_biasGradientPart1, 4, sizeof (unsigned int), &output_maps_);

    if (error != CL_SUCCESS) {
      FATAL ("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = { output_maps_, samples };

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_biasGradientPart1, 2, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);

    if (error != CL_SUCCESS) {
      FATAL ("Error enqueueing kernel: " << (signed int) error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (CLHelper::queue);

    if (error != CL_SUCCESS) {
      FATAL ("Error finishing command queue: " << (signed int) error);
    }

#endif

  }

  {
    cl_uint error = 0;

    bias_->delta.MoveToGPU (true);
    const unsigned int samples = input_->data.samples();
    error |= clSetKernelArg (CLHelper::k_biasGradientPart2, 0, sizeof (cl_mem), &bias_buffer_.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_biasGradientPart2, 1, sizeof (cl_mem), &bias_->delta.cl_data_ptr_);
    error |= clSetKernelArg (CLHelper::k_biasGradientPart2, 2, sizeof (unsigned int), &output_maps_);
    error |= clSetKernelArg (CLHelper::k_biasGradientPart2, 3, sizeof (unsigned int), &samples);
    error |= clSetKernelArg (CLHelper::k_biasGradientPart2, 4, sizeof (datum), &one);

    if (error != CL_SUCCESS) {
      FATAL ("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = { output_maps_ };

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_biasGradientPart2, 1, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);

    if (error != CL_SUCCESS) {
      FATAL ("Error enqueueing kernel: " << (signed int) error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (CLHelper::queue);

    if (error != CL_SUCCESS) {
      FATAL ("Error finishing command queue: " << (signed int) error);
    }

#endif

  }
#else
  bias_->delta.Clear();

  #pragma omp parallel for default(shared)

  for (unsigned int omap = 0; omap < output_maps_; omap++) {
    for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
//      for (unsigned int imap = 0; imap < input_maps_; imap++) {
      for (unsigned int doy = 0; doy < output_height_; doy++) {
        for (unsigned int dox = 0; dox < output_width_; dox++) {
          const datum doval =
            *output_->delta.data_ptr_const (dox, doy, omap, sample);
          bias_->delta[omap] += doval; //* 2.0f;
        }
      }

//      }
    }
  }

#endif
}

void ConvolutionLayer::OnLayerConnect (Layer* next_layer) {
  unsigned int next_layer_gain = next_layer->Gain();
  unsigned int this_layer_gain = Gain();

  const datum range = sqrt (6) / sqrt (next_layer_gain + this_layer_gain);

#ifdef BUILD_OPENCL
  weights_->data.MoveToCPU();
#endif
  std::uniform_real_distribution<datum> dist_weights (-range , range);

  for (std::size_t i = 0; i < weights_->data.elements(); i++) {
    weights_->data[i] = dist_weights (rand_);
  }

  LOGDEBUG << "Updating weights: " << this_layer_gain << " -> "
           << next_layer_gain;
}

void ConvolutionLayer::im2colff() {
  #pragma omp parallel for default(shared)

  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int imap = 0; imap < input_maps_; imap++) {
      for (unsigned int oy = 0; oy < output_height_; oy++) {
        for (unsigned int ox = 0; ox < output_width_; ox++) {
          // Maybe putting y on the outside will help with large kernels
          // and short cache lines?
          for (unsigned int ky = 0; ky < kernel_height_; ky++) {
            unsigned int iy = oy + ky;
            const datum* source = input_->data.data_ptr_const (ox, iy, imap, sample);
            datum* target = im2col_ff_buffer.data_ptr (kernel_width_ * ky,
                            imap, oy *
                            output_width_ +
                            ox, sample);

            std::memcpy (target, source, sizeof (datum) * kernel_width_);
          }
        }
      }
    }
  }
}

void ConvolutionLayer::col2imff() {
  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int omap = 0; omap < output_maps_; omap++) {
      Tensor::CopyMap (ff_output_buffer, omap, sample, output_->data,
                       sample, omap);
    }
  }
}

void ConvolutionLayer::im2colbp() {
  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int omap = 0; omap < output_maps_; omap++) {
      Tensor::CopyMap (output_->delta, sample, omap, bp_deltay_buffer,
                       omap, sample);
    }
  }
}

void ConvolutionLayer::col2imbp() {
  #pragma omp parallel for default(shared)

  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int imap = 0; imap < input_maps_; imap++) {
      for (unsigned int dxx = 0; dxx < output_width_; dxx++) {
        for (unsigned int dxy = 0; dxy < output_height_; dxy++) {
          for (unsigned int ky = 0; ky < kernel_height_; ky++) {
            const datum* kernel_line = bp_deltax_buffer.data_ptr_const (
                                         kernel_width_ * ky, imap, dxy * output_width_ + dxx, sample);
            datum* target_line = input_->delta.data_ptr (
                                   dxx, dxy + ky, imap, sample);

            for (unsigned int kx = 0; kx < kernel_width_; kx++) {
              target_line[kx] += kernel_line[kx];
            }
          }
        }
      }
    }
  }
}

bool ConvolutionLayer::IsOpenCLAware() {
#ifdef BUILD_OPENCL_CONV
  return true;
#else
  return false;
#endif
}

}

