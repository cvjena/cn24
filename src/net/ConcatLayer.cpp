/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <cstring>

#include "ConcatLayer.h"

namespace Conv {


ConcatLayer::ConcatLayer() {
  LOGDEBUG << "Instance created.";
}

bool ConcatLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
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
  
  if(input_a->data.height() != 1 || input_a->data.maps() != 1 ||
    input_b->data.height() != 1 || input_b->data.maps() != 1) {
    LOGERROR << "Input tensors need to be flat!";
    return false;
  }
  
  if(input_a->data.samples() != input_b->data.samples()) {
    LOGERROR << "Sample count doesn't match!";
    return false;
  }
  
  unsigned int width_a = input_a->data.width();
  unsigned int width_b = input_b->data.width();
  unsigned int samples = input_a->data.samples();
  CombinedTensor* output = new CombinedTensor(samples, width_a + width_b);
  
  outputs.push_back(output);
  return true;
}

bool ConcatLayer::Connect (const std::vector< CombinedTensor* >& inputs,
                           const std::vector< CombinedTensor* >& outputs) {
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
  
  if(input_a->data.height() != 1 || input_a->data.maps() != 1 ||
    input_b->data.height() != 1 || input_b->data.maps() != 1 ||
    output->data.height() != 1 || output->data.maps() != 1) {
    LOGERROR << "Tensors need to be flat!";
    return false;
  }
  
  if(input_a->data.samples() != input_b->data.samples()) {
    LOGERROR << "Sample count doesn't match!";
    return false;
  }
  
  if(output->data.width() != (input_a->data.width() + input_b->data.width())) {
    LOGERROR << "Wrong output dimensions!";
    return false;
  }
  
  width_a_ = input_a->data.width();
  width_b_ = input_b->data.width();
  samples_ = input_a->data.samples();
  
  input_a_ = input_a;
  input_b_ = input_b;
  output_ = output;
  
  return true;
}

void ConcatLayer::FeedForward() {
#pragma omp parallel for default(shared)
  for(unsigned int s = 0; s < samples_; s++) {
    const datum* src_a = input_a_->data.data_ptr_const(0,0,0,s);
    const datum* src_b = input_b_->data.data_ptr_const(0,0,0,s);
    
    datum* tgt_a = output_->data.data_ptr(0,0,0,s);
    datum* tgt_b = output_->data.data_ptr(width_a_,0,0,s);
    
    std::memcpy(tgt_a, src_a, width_a_ * sizeof(datum)/sizeof(char));
    std::memcpy(tgt_b, src_b, width_b_ * sizeof(datum)/sizeof(char));
  }
}

void ConcatLayer::BackPropagate() {
#pragma omp parallel for default(shared)
  for(unsigned int s = 0; s < samples_; s++) {
    const datum* src_a = output_->delta.data_ptr_const(0,0,0,s);
    const datum* src_b = output_->delta.data_ptr_const(width_a_,0,0,s);
    
    datum* tgt_a = input_a_->delta.data_ptr(0,0,0,s);
    datum* tgt_b = input_b_->delta.data_ptr(0,0,0,s);
    
    std::memcpy(tgt_a, src_a, width_a_ * sizeof(datum)/sizeof(char));
    std::memcpy(tgt_b, src_b, width_b_ * sizeof(datum)/sizeof(char));
  }
}

}