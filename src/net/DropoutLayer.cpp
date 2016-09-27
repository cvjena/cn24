/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <vector>
#include <random>
#include <algorithm>


#include "CLHelper.h"
#include "CombinedTensor.h"
#include "Log.h"
#include "BoundingBox.h"

#include "DropoutLayer.h"
#include "Test.h"

namespace Conv {

DropoutLayer::DropoutLayer(JSON configuration): SimpleLayer(configuration) {
  unsigned int seed = 0;
  if(configuration.count("dropout_fraction") != 1 || !configuration["dropout_fraction"].is_number()) {
    FATAL("Dropout fraction missing!");
  }
  dropout_fraction_ = configuration["dropout_fraction"];
  AssertGreater((datum)0, dropout_fraction_, "Dropout fraction");
  AssertLess((datum)1, dropout_fraction_, "Dropout fraction");

  scale_ = 1.0 / (1.0 - dropout_fraction_);

  if(configuration.count("seed") == 1 && configuration["seed"].is_number()) {
		seed = configuration["seed"];
	}
  seed_ = seed;
  rand_.seed(seed);

  LOGDEBUG << "Initialized. Dropout fraction: " << dropout_fraction_;
}
  
bool DropoutLayer::CreateOutputs (
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

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      input->data.width(),
      input->data.height(),
      input->data.maps());

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool DropoutLayer::Connect (const CombinedTensor* input,
                                 CombinedTensor* output) {
  // Check dimensions for equality
  bool valid = input->data.samples() == output->data.samples() &&
               input->data.width() == output->data.width() &&
               input->data.height() == output->data.height() &&
               input->data.maps() == output->data.maps();

  if(valid)
    dropout_mask_.Resize(input->data);

  return valid;
}

void DropoutLayer::FeedForward() {
  if(!net_->IsTesting() || net_->IsGradientTesting()) {
    if(net_->IsGradientTesting())
      rand_.seed(seed_);
#ifdef BUILD_OPENCL
    dropout_mask_.MoveToCPU(true);
#endif
    datum* mask_ptr = dropout_mask_.data_ptr();

    std::bernoulli_distribution dist = std::bernoulli_distribution(1.0 - dropout_fraction_);
    for (unsigned int element = 0; element < input_->data.elements(); element++) {
      if(dist(rand_)) {
        mask_ptr[element] = scale_;
      } else {
        mask_ptr[element] = 0;
      }
    }

#ifdef BUILD_OPENCL
    input_->data.MoveToGPU();
    dropout_mask_.MoveToGPU();
    output_->data.MoveToGPU(true);
    cl_int error = 0;

    error |= clSetKernelArg(CLHelper::k_applyMask, 0, sizeof(cl_mem), &(input_->data.cl_data_ptr_));
    error |= clSetKernelArg(CLHelper::k_applyMask, 1, sizeof(cl_mem), &(dropout_mask_.cl_data_ptr_));
    error |= clSetKernelArg(CLHelper::k_applyMask, 2, sizeof(cl_mem), &(output_->data.cl_data_ptr_));

    if (error != CL_SUCCESS) {
      FATAL("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = {(size_t)input_->data.elements()};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_applyMask, 1, NULL,
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
    const datum* in_data_ptr = input_->data.data_ptr_const();
    datum* out_data_ptr = output_->data.data_ptr();
#pragma omp parallel for default(shared)
    for (unsigned int element = 0; element < input_->data.elements(); element++) {
      out_data_ptr[element] = in_data_ptr[element] * mask_ptr[element];
    }
#endif
  } else {
    Tensor::Copy(input_->data, output_->data);
  }
}

void DropoutLayer::BackPropagate() {
  if(!net_->IsTesting() || net_->IsGradientTesting()) {
#ifdef BUILD_OPENCL
    output_->delta.MoveToGPU();
    dropout_mask_.MoveToGPU();
    input_->delta.MoveToGPU(true);
    cl_int error = 0;

    error |= clSetKernelArg(CLHelper::k_applyMask, 0, sizeof(cl_mem), &(output_->delta.cl_data_ptr_));
    error |= clSetKernelArg(CLHelper::k_applyMask, 1, sizeof(cl_mem), &(dropout_mask_.cl_data_ptr_));
    error |= clSetKernelArg(CLHelper::k_applyMask, 2, sizeof(cl_mem), &(input_->delta.cl_data_ptr_));

    if (error != CL_SUCCESS) {
      FATAL("Error setting kernel args: " << (signed int) error);
    }

    size_t global_work_size[] = {(size_t)output_->data.elements()};

    error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_applyMask, 1, NULL,
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
    datum* in_delta_ptr = input_->delta.data_ptr();
    const datum* out_delta_ptr = output_->delta.data_ptr_const();
    const datum* mask_ptr = dropout_mask_.data_ptr_const();
#pragma omp parallel for default(shared)
    for (unsigned int element = 0; element < input_->data.elements(); element++) {
      in_delta_ptr[element] = out_delta_ptr[element] * mask_ptr[element];
    }
#endif
  } else {
    // Except when gradient testing, this should never happen
    Tensor::Copy(output_->delta, input_->delta);
  }
}


}
