/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
* @file ActivationFunctions.cpp
* @brief This file contains implementations for all the NonLinearityLayers.
* @see NonLinearityLayer.h for declarations, specifically the NL_LAYER
*      macros.
*
* @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
*/

#include <cmath>


#include "CLHelper.h"
#include "NonLinearityLayer.h"

namespace Conv {

bool SigmoidLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL
  return true;
#else
  return false;
#endif
}

bool TanhLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL
  return true;
#else
  return false;
#endif
}

bool LeakyReLULayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL
  return true;
#else
  return false;
#endif
}

void SigmoidLayer::FeedForward () {
#ifdef BUILD_OPENCL
  cl_uint error = 0;
  input_->data.MoveToGPU ();
  output_->data.MoveToGPU ();

  error |= clSetKernelArg (CLHelper::k_nlSigm, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlSigm, 1, sizeof (cl_mem), &output_->data.cl_data_ptr_);

  if (error != CL_SUCCESS) {
    FATAL("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = {input_->data.elements ()};

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_nlSigm, 1, NULL,
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

#else
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const() [element];

    // Calculate sigmoid: sigm(x) = 1.0 / (1.0 + e^-x)
    const datum output_data = 1.0 / (1.0 + exp (-input_data));
    output_->data.data_ptr() [element] = output_data;
  }
#endif
}

void SigmoidLayer::BackPropagate () {
#ifdef BUILD_OPENCL
  cl_uint error = 0;
  
  input_->delta.MoveToGPU ();
  output_->data.MoveToGPU ();
  output_->delta.MoveToGPU ();
  
  error |= clSetKernelArg (CLHelper::k_nlSigmBackward, 0, sizeof (cl_mem), &input_->delta.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlSigmBackward, 1, sizeof (cl_mem), &output_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlSigmBackward, 2, sizeof (cl_mem), &output_->delta.cl_data_ptr_);

  if (error != CL_SUCCESS) {
    FATAL("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = {input_->data.elements ()};

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_nlSigmBackward, 1, NULL,
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

 
  
#else
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements (); element++) {
    const datum output_delta = output_->delta.data_ptr_const ()[element];
    const datum output_data = output_->data.data_ptr_const ()[element];

    // sigm'(x) = sigm(x) * (1.0 - sigm(x))
    // sigm(x) = output
    // this is why we use output_data here (so we don't need to calculate
    // sigm(x) twice).
    // This may be slower for wide networks and large batches
    // because of cache limitations.

    const datum input_delta = (datum) (output_delta * output_data * (1.0 - output_data));
    input_->delta.data_ptr ()[element] = input_delta;
  }
#endif
}

void TanhLayer::FeedForward () {
#ifdef BUILD_OPENCL
  cl_uint error = 0;
  input_->data.MoveToGPU ();
  output_->data.MoveToGPU ();

  error |= clSetKernelArg (CLHelper::k_nlTanh, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlTanh, 1, sizeof (cl_mem), &output_->data.cl_data_ptr_);

  if (error != CL_SUCCESS) {
    FATAL("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = {input_->data.elements ()};

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_nlTanh, 1, NULL,
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

#else
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const() [element];

    // Calculate hyperbolic tangent: tanh(x) = 1.0 - 2.0 / (e^(2*x) + 1)
    const datum output_data = 1.0 - 2.0 / (exp (2.0 * input_data) + 1.0);
    output_->data.data_ptr() [element] = output_data;
  }
#endif
}

void TanhLayer::BackPropagate () {
#ifdef BUILD_OPENCL
  cl_uint error = 0;
  
  input_->delta.MoveToGPU ();
  output_->data.MoveToGPU ();
  output_->delta.MoveToGPU ();
  
  error |= clSetKernelArg (CLHelper::k_nlTanhBackward, 0, sizeof (cl_mem), &input_->delta.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlTanhBackward, 1, sizeof (cl_mem), &output_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlTanhBackward, 2, sizeof (cl_mem), &output_->delta.cl_data_ptr_);

  if (error != CL_SUCCESS) {
    FATAL("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = {input_->data.elements ()};

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_nlTanhBackward, 1, NULL,
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

  
#else
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements (); element++) {
    const datum output_delta = output_->delta.data_ptr_const ()[element];
    const datum output_data = output_->data.data_ptr_const ()[element];

    // tanh'(x) = 1 - (tanh(x))^2
    // tanh(x) = output
    // see SigmoidLayer::BackPropagate for an explanation

    const datum input_delta = (datum) (output_delta * (1.0 - output_data * output_data));
    input_->delta.data_ptr ()[element] = input_delta;
  }
#endif
}

void ReLULayer::FeedForward () {
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements (); element++) {
    const datum input_data = input_->data.data_ptr_const ()[element];

    // max(0, x)
    const datum output_data = input_data > 0 ? input_data : 0;
    output_->data.data_ptr ()[element] = output_data;
  }
}

void ReLULayer::BackPropagate () {
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements (); element++) {
    const datum output_delta = output_->delta.data_ptr_const ()[element];
    const datum input_data = input_->data.data_ptr_const ()[element];

    // There is more than one way to do this. max(0,x) is not differentiable
    // at x=0 so we have to make a choice. It doesn't affect the learning in
    // any meaningful way.
    const datum input_delta = (input_data > 0 ? output_delta : 0);
    input_->delta.data_ptr ()[element] = input_delta;
  }
}

void SoftmaxLayer::FeedForward () {
#pragma omp parallel for default(shared)
  for (std::size_t sample = 0; sample < input_->data.samples (); sample++) {
    float sum = 0.0f;
    for(std::size_t element = 0; element < (input_->data.elements() / input_->data.samples()); element++) {
      sum += exp (*input_->data.data_ptr (element,0,0,sample));
    }
    for (std::size_t element = 0; element < (input_->data.elements() / input_->data.samples()) ; element++) {
      *output_->data.data_ptr(element,0,0,sample) =
          (datum) exp (*input_->data.data_ptr (element,0,0,sample)) / sum;
    }
  }
}

void SoftmaxLayer::BackPropagate () {
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements (); element++) {
    const datum output_delta = output_->delta.data_ptr_const ()[element];
    input_->delta.data_ptr ()[element] = output_delta;
  }
}

void LeakyReLULayer::FeedForward () {
#ifdef BUILD_OPENCL
  cl_uint error = 0;
  input_->data.MoveToGPU ();
  output_->data.MoveToGPU ();

  error |= clSetKernelArg (CLHelper::k_nlLeaky, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlLeaky, 1, sizeof (cl_mem), &output_->data.cl_data_ptr_);

  if (error != CL_SUCCESS) {
    FATAL("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = {input_->data.elements ()};

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_nlLeaky, 1, NULL,
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

#else
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const ()[element];

    // max(0, x)
    const datum output_data = input_data > 0 ? input_data : 0.1 * input_data;
    output_->data.data_ptr ()[element] = output_data;
  }
#endif
}

void LeakyReLULayer::BackPropagate () {
#ifdef BUILD_OPENCL
  cl_uint error = 0;

  input_->delta.MoveToGPU ();
  input_->data.MoveToGPU ();
  output_->delta.MoveToGPU ();

  error |= clSetKernelArg (CLHelper::k_nlLeakyBackward, 0, sizeof (cl_mem), &input_->delta.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlLeakyBackward, 1, sizeof (cl_mem), &input_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_nlLeakyBackward, 2, sizeof (cl_mem), &output_->delta.cl_data_ptr_);

  if (error != CL_SUCCESS) {
    FATAL("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = {input_->data.elements ()};

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_nlLeakyBackward, 1, NULL,
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



#else
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements (); element++) {
    const datum output_delta = output_->delta.data_ptr_const ()[element];
    const datum input_data = input_->data.data_ptr_const ()[element];

    // There is more than one way to do this. max(0,x) is not differentiable
    // at x=0 so we have to make a choice. It doesn't affect the learning in
    // any meaningful way.
    const datum input_delta = (input_data > 0 ? output_delta : ((datum)0.1) * output_delta);
    input_->delta.data_ptr ()[element] = input_delta;
  }
#endif
}


}
