/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <random>
#include <cmath>

#ifdef BUILD_OPENCL
#include <CL/cl.h>
#include "Init.h"
#endif

#include "Config.h"
#include "Log.h"
#include "FullyConnectedLayer.h"

namespace Conv {

FullyConnectedLayer::FullyConnectedLayer (const unsigned int neurons,
    const int seed) :
  neurons_ (neurons), rand_ (seed) {
  if (neurons_ < 1) {
    FATAL ("Less than one neuron requested. This is not a copy constructor!");
    return;
  }

#ifndef BUILD_BLAS
  LOGDEBUG << "Using slow mode (no BLAS).";
#endif

  LOGDEBUG << "Instance created: " << neurons_ << " neurons";

  // See ConvolutionLayer constructor for explanation
  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }
}

bool FullyConnectedLayer::CreateOutputs (
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
  if (input->data.height() != 1 || input->data.maps() != 1) {
    LOGERROR << "Unsupported input dimensions " << input->data;
    return false;
  }

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(), neurons_);

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool FullyConnectedLayer::Connect (const CombinedTensor* input,
                                   CombinedTensor* output) {
  // Validate CombinedTensor dimensions
  bool valid = input->data.height() == 1 && output->data.height() == 1 &&
               input->data.maps() == 1 && output->data.maps() == 1 &&
               input->data.samples() == output->data.samples() &&
               output->data.width() == neurons_;

  if (!valid)
    return false;

  // Create weight and bias tensors
  input_units_ = input->data.width();
  weights_ = new CombinedTensor (1, neurons_, input_units_);
  bias_ = new CombinedTensor (1, neurons_);

  // Initialize weights
  weights_->data.Clear();

  // Initialize biases
  bias_->data.Clear (1.0);

  // There may be a better way to do this...
  ones_ = new datum [input->data.samples()];
  for (unsigned int i = 0; i < input->data.samples(); i++)
    ones_[i] = 1;

  parameters_.push_back (weights_);
  parameters_.push_back (bias_);
  return true;
}

void FullyConnectedLayer::FeedForward() {
#ifdef BUILD_OPENCL
  cl_int error = 0;
  input_->data.MoveToGPU();
  weights_->data.MoveToGPU();
  bias_->data.MoveToGPU();
  output_->data.MoveToGPU(true);

  error |= clSetKernelArg (System::k_biasedMatrixVector, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
  error |= clSetKernelArg (System::k_biasedMatrixVector, 1, sizeof (cl_mem), &weights_->data.cl_data_ptr_);
  error |= clSetKernelArg (System::k_biasedMatrixVector, 2, sizeof (cl_mem), &bias_->data.cl_data_ptr_);
  error |= clSetKernelArg (System::k_biasedMatrixVector, 3, sizeof (cl_mem), &output_->data.cl_data_ptr_);
  error |= clSetKernelArg (System::k_biasedMatrixVector, 4, sizeof (unsigned int), &input_units_);
  error |= clSetKernelArg (System::k_biasedMatrixVector, 5, sizeof (unsigned int), &neurons_);
  error |= clSetKernelArg (System::k_biasedMatrixVector, 6, sizeof (datum), &weight_factor_);

  if (error != CL_SUCCESS) {
    FATAL ("Error setting kernel args: " << error);
  }

  size_t global_work_size[] = { neurons_, input_->data.samples() };

  error = clEnqueueNDRangeKernel (System::queue, System::k_biasedMatrixVector, 2, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    FATAL ("Error enqueueing kernel: " << error);
  }

#ifdef BRUTAL_FINISH
  error = clFinish (System::queue);
  if (error != CL_SUCCESS) {
    FATAL ("Error finishing command queue: " << error);
  }
#endif

#else
#ifdef BUILD_BLAS
  // (rows * cols) W (input_units_ x neurons_)
  // (rows * cols) X (samples_ x input_units)
  // (rows * cols) Y (samples_ x neurons_)
  // b (neurons_)
  const datum* W = weights_->data.data_ptr_const();
  const datum* X = input_->data.data_ptr_const();
  const datum* b = bias_->data.data_ptr_const();
  datum* Y = output_->data.data_ptr();

  // calculate samples
  GEMM (CblasRowMajor, CblasNoTrans, CblasNoTrans, input_->data.samples(), neurons_,
        input_units_, weight_factor_, X, input_units_, W, neurons_, 0.0, Y, neurons_);

  // add bias term
  GEMM (CblasRowMajor, CblasNoTrans, CblasNoTrans, input_->data.samples(), neurons_,
        1, 1.0, ones_, 1, b, neurons_, 1.0, Y, neurons_);
#else
  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int i = 0; i < neurons_; i++) {
      datum neuron_sum = 0;

      // Add input times weight
      for (unsigned int j = 0; j < input_units_; j++) {
        const datum w = * (weights_->data.data_ptr_const (i, j));
        const datum x = * (input_->data.data_ptr_const (j, 0, 0, sample));
        neuron_sum += weight_factor_ * w * x;
      }

      // Add bias term
      datum bias = * (bias_->data.data_ptr_const (i));

      * (output_->data.data_ptr (i, 0, 0, sample)) = neuron_sum + bias;
    }
  }
#endif // else BUILD_BLAS
#endif // else BUILD_OPENCL
}

void FullyConnectedLayer::BackPropagate() {
  static datum one = 1.0;
  weights_->delta.Clear();
  bias_->delta.Clear();
  input_->delta.Clear();
#ifdef BUILD_OPENCL
  cl_int error = 0;
  const unsigned int samples = input_->data.samples();

  input_->data.MoveToGPU();
  input_->delta.MoveToGPU();
  weights_->delta.MoveToGPU();
  bias_->delta.MoveToGPU();
  output_->delta.MoveToGPU();

  {
    // Backpropagate
    error |= clSetKernelArg (System::k_biasedMatrixVectorBackward, 0, sizeof (cl_mem), &input_->delta.cl_data_ptr_);
    error |= clSetKernelArg (System::k_biasedMatrixVectorBackward, 1, sizeof (cl_mem), &weights_->data.cl_data_ptr_);
    error |= clSetKernelArg (System::k_biasedMatrixVectorBackward, 2, sizeof (cl_mem), &output_->delta.cl_data_ptr_);
    error |= clSetKernelArg (System::k_biasedMatrixVectorBackward, 3, sizeof (unsigned int), &input_units_);
    error |= clSetKernelArg (System::k_biasedMatrixVectorBackward, 4, sizeof (unsigned int), &neurons_);

    if (error != CL_SUCCESS) {
      FATAL ("Error setting kernel args: " << error);
    }

    size_t global_work_size[] = { input_units_, input_->data.samples() };

    error = clEnqueueNDRangeKernel (System::queue, System::k_biasedMatrixVectorBackward, 2, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
      FATAL ("Error enqueueing kernel: " << error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (System::queue);
    if (error != CL_SUCCESS) {
      FATAL ("Error finishing command queue: " << error);
    }
#endif
  }

  // Calculate gradient X * dY = dW
  {
    error |= clSetKernelArg (System::k_matrixMatrix, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
    error |= clSetKernelArg (System::k_matrixMatrix, 1, sizeof (cl_mem), &weights_->delta.cl_data_ptr_);
    error |= clSetKernelArg (System::k_matrixMatrix, 2, sizeof (cl_mem), &output_->delta.cl_data_ptr_);
    error |= clSetKernelArg (System::k_matrixMatrix, 3, sizeof (unsigned int), &input_units_);
    error |= clSetKernelArg (System::k_matrixMatrix, 4, sizeof (unsigned int), &neurons_);
    error |= clSetKernelArg (System::k_matrixMatrix, 5, sizeof (unsigned int), &samples);
    error |= clSetKernelArg (System::k_matrixMatrix, 6, sizeof (datum), &one);

    if (error != CL_SUCCESS) {
      FATAL ("Error setting kernel args: " << error);
    }

    size_t global_work_size[] = { neurons_, input_units_ };

    error = clEnqueueNDRangeKernel (System::queue, System::k_matrixMatrix, 2, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
      FATAL ("Error enqueueing kernel: " << error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (System::queue);
    if (error != CL_SUCCESS) {
      FATAL ("Error finishing command queue: " << error);
    }
#endif

  }

  // Calculate bias gradient
  {
    error |= clSetKernelArg (System::k_biasedMatrixVectorGrad, 0, sizeof (cl_mem), &bias_->delta.cl_data_ptr_);
    error |= clSetKernelArg (System::k_biasedMatrixVectorGrad, 1, sizeof (cl_mem), &output_->delta.cl_data_ptr_);
    error |= clSetKernelArg (System::k_biasedMatrixVectorGrad, 2, sizeof (unsigned int), &neurons_);
    error |= clSetKernelArg (System::k_biasedMatrixVectorGrad, 3, sizeof (unsigned int), &samples);
    error |= clSetKernelArg (System::k_biasedMatrixVectorGrad, 4, sizeof (datum), &one);

    if (error != CL_SUCCESS) {
      FATAL ("Error setting kernel args: " << error);
    }

    size_t global_work_size[] = { neurons_ };

    error = clEnqueueNDRangeKernel (System::queue, System::k_biasedMatrixVectorGrad, 1, NULL,
                                    global_work_size, NULL, 0, NULL, NULL);
    if (error != CL_SUCCESS) {
      FATAL ("Error enqueueing kernel: " << error);
    }

#ifdef BRUTAL_FINISH
    error = clFinish (System::queue);
    if (error != CL_SUCCESS) {
      FATAL ("Error finishing command queue: " << error);
    }
#endif
  }

#else
#ifdef BUILD_BLAS
  const datum* dY = output_->delta.data_ptr_const();
  const datum* X = input_->data.data_ptr_const();
  const datum* W = weights_->data.data_ptr_const();
  datum* dW = weights_->delta.data_ptr();
  datum* dX = input_->delta.data_ptr();
  datum* db = bias_->delta.data_ptr();

  // Backpropagate
  if (backprop_enabled_) {
    GEMM (CblasRowMajor, CblasNoTrans, CblasTrans, input_->data.samples(), input_units_,
          neurons_, 1.0, dY, neurons_, W, neurons_, 0.0, dX, input_units_);
  }

  // Calculate gradient
  GEMM (CblasRowMajor, CblasTrans, CblasNoTrans, input_units_, neurons_, input_->data.samples(),
        one, X, input_units_, dY, neurons_, 0.0, dW, neurons_);

  // Bias gradient
  GEMM (CblasRowMajor, CblasNoTrans, CblasNoTrans, 1, neurons_, input_->data.samples(),
        one, ones_, input_->data.samples(), dY, neurons_, 0.0, db, neurons_);

#else
  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int j = 0; j < input_units_; j++) {
      datum input_delta = 0;

      for (unsigned int i = 0; i < neurons_; i++) {
        const datum output_delta = * (output_->delta.data_ptr_const (i, 0, 0, sample));

        // Backpropagate
        if (backprop_enabled_)
          input_delta += output_delta * * (weights_->data.data_ptr_const (i, j));

        // Calculate gradient
        const datum input_data = * (input_->data.data_ptr_const (j, 0, 0, sample));

        * (weights_->delta.data_ptr (i, j)) += input_data * output_delta;
      }

      if (backprop_enabled_)
        * (input_->delta.data_ptr (j, 0, 0, sample)) = input_delta;
    }


    // Bias gradient here
    for (unsigned int i = 0; i < neurons_; i++) {
      const datum output_delta = * (output_->delta.data_ptr_const (i, 0, 0, sample));
      * (bias_->delta.data_ptr (i)) += output_delta;
    }
  }
#endif
#endif
}

void FullyConnectedLayer::OnLayerConnect (Layer* next_layer) {
  unsigned int next_layer_gain = next_layer->Gain();
  unsigned int this_layer_gain = Gain();

  const datum range = sqrt (6) / sqrt (next_layer_gain + this_layer_gain);

  std::uniform_real_distribution<datum> dist_weights (-range , range);
  for (std::size_t i = 0; i < weights_->data.elements(); i++) {
    weights_->data[i] = dist_weights (rand_);
  }

  LOGDEBUG << "Updating weights: " << this_layer_gain << " -> "
           << next_layer_gain;
}



}
