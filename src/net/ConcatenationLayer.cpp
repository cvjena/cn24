/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cstring>

#include "ConcatenationLayer.h"

namespace Conv {


ConcatenationLayer::ConcatenationLayer() : Layer(JSON::object()) {
  LOGDEBUG << "Instance created.";
}

bool ConcatenationLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                                 std::vector< CombinedTensor* >& outputs) {
  if(inputs.size() != 2) {
    LOGERROR << "Needs two inputs!";
    return false;
  }
  
  CombinedTensor* input_a = inputs[0];
  CombinedTensor* input_b = inputs[1];
  
  if(input_a == nullptr || input_b == nullptr) {
    LOGERROR << "Null pointer supplied";
    return false;
  }
  
  if(input_a->data.width() != input_b->data.width()
    && input_a->data.height() != input_b->data.height()) {
    LOGERROR << "Dimensions don't match!";
  }
  
  if(input_a->data.samples() != input_b->data.samples()) {
    LOGERROR << "Sample count doesn't match!";
    return false;
  }
  
  unsigned int maps_a = input_a->data.maps();
  unsigned int maps_b = input_b->data.maps();
  unsigned int samples = input_a->data.samples();
  CombinedTensor* output = new CombinedTensor(samples, input_a->data.width(),
    input_b->data.height(), maps_a + maps_b);
  
  outputs.push_back(output);
  return true;
}

bool ConcatenationLayer::Connect (const std::vector< CombinedTensor* >& inputs,
                           const std::vector< CombinedTensor* >& outputs,
                           const NetStatus* status ) {
  UNREFERENCED_PARAMETER(status);
  if(inputs.size() != 2) {
    LOGERROR << "Needs two inputs!";
    return false;
  }
  
  if(outputs.size() != 1) {
    LOGERROR << "Needs exactly one output!";
    return false;
  }
  
  CombinedTensor* input_a = inputs[0];
  CombinedTensor* input_b = inputs[1];
  CombinedTensor* output = outputs[0];
  
  if(input_a == nullptr || input_b == nullptr || output == nullptr) {
    LOGERROR << "Null pointer supplied";
    return false;
  }
  
  if(input_a->data.samples() != input_b->data.samples()) {
    LOGERROR << "Sample count doesn't match!";
    return false;
  }
  
  if(output->data.elements() != (input_a->data.elements() + input_b->data.elements())) {
    LOGERROR << "Wrong output dimensions!";
    return false;
  }
  
  maps_a_ = input_a->data.maps();
  maps_b_ = input_b->data.maps();
  samples_ = input_a->data.samples();
  
  input_a_ = input_a;
  input_b_ = input_b;
  output_ = output;
  
  return true;
}

void ConcatenationLayer::FeedForward() {
#pragma omp parallel for default(shared)
  for(unsigned int s = 0; s < samples_; s++) {
    for(unsigned int m = 0; m < maps_a_; m++)
      Tensor::CopyMap(input_a_->data, s, m ,output_->data, s, m);
    
    for(unsigned int m = 0; m < maps_b_; m++)
      Tensor::CopyMap(input_b_->data, s, m, output_->data, s, maps_a_ + m);
  }
}

void ConcatenationLayer::BackPropagate() {
#pragma omp parallel for default(shared)
  for(unsigned int s = 0; s < samples_; s++) {
    for(unsigned int m = 0; m < maps_a_; m++)
      Tensor::CopyMap(output_->delta, s, m, input_a_->delta, s, m);
    
    for(unsigned int m = 0; m < maps_b_; m++)
      Tensor::CopyMap(output_->delta, s, maps_a_ + m, input_b_->delta, s, m);
  }
}

}
