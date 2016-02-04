/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cmath>

#include "CombinedTensor.h"

#include "HMaxActivationFunction.h"

namespace Conv {
bool HMaxActivationFunction::CreateOutputs (
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

bool HMaxActivationFunction::Connect (const CombinedTensor* input,
                                 CombinedTensor* output) {
  // Check dimensions for equality
  bool valid = input->data.samples() == output->data.samples() &&
  input->data.width() == output->data.width() &&
  input->data.height() == output->data.height() &&
  input->data.maps() == output->data.maps();
  
  if(!valid)
    return false;
  
  weights_ = new CombinedTensor(1, 2, 1, 1);
  weights_->data.Clear(1.0);
  weights_->delta.Clear();
  
  parameters_.push_back(weights_);
  
  return true;
}
  
void HMaxActivationFunction::FeedForward() {
  const datum a = *(weights_->data.data_ptr(0));
  const datum b = *(weights_->data.data_ptr(1));
  
  LOGDEBUG << "a: " << a << ", b:" << b;
  
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const() [element];

    // Calculate sigmoid function
    const datum output_data = 1.0 / 1.0 + exp(-(a * input_data + b));
    output_->data.data_ptr() [element] = output_data;
  }
}
  
void HMaxActivationFunction::BackPropagate() {
  // Calculate gradient w.r.t. input
  const datum a = weights_->data.data_ptr_const()[0];
  const datum b = weights_->data.data_ptr_const()[1];
  
  datum delta_a = -((datum)(input_->data.elements()))/a;
  datum delta_b = -(datum)(input_->data.elements());
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const() [element];
    const datum output_data = output_->data.data_ptr_const() [element];
    const datum output_delta = output_->delta.data_ptr_const ()[element];

    // Calculate derivative
    const datum input_delta = output_delta *  a * exp(-(a * input_data + b)) * output_data * output_data;
    input_->delta.data_ptr() [element] = input_delta;
    
    // Calculate gradient of KL-divergence w.r.t. weights
    delta_a += -input_data + (2.0 + 1.0/mu_) * input_data * output_data - (1.0/mu_) * input_data * output_data * output_data;
    delta_b += (2.0 + 1.0/mu_) * output_data - (1.0/mu_) * output_data * output_data;
  } 
  
  weights_->delta.data_ptr()[0] = local_lr_ * delta_a;
  weights_->delta.data_ptr()[1] = local_lr_ * delta_b;
  LOGDEBUG << "delta a: " << delta_a << ", delta b:" << delta_b;
}


}