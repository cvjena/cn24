/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <limits>

#include "Log.h"
#include "LocalResponseNormalizationLayer.h"

namespace Conv {

LocalResponseNormalizationLayer::
  LocalResponseNormalizationLayer(const unsigned int size,
    const datum alpha, const datum beta,
    const LocalResponseNormalizationLayer::NormalizationMethod normalization_method) :
    size_(size), alpha_(alpha), beta_(beta), normalization_method_(normalization_method) {
  LOGDEBUG << "Instance created, size: " << size_ << ", alpha: " << alpha_ 
  << ", beta: " << beta_ << ", method: " << (normalization_method_ == ACROSS_CHANNELS) ? "across" : "within";

}

bool LocalResponseNormalizationLayer::
  CreateOutputs(const std::vector< CombinedTensor* >& inputs,
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
  
  // Create ouput
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
    input->data.width(), input->data.height(), input->data.maps());
  
  // Tell network about the output
  outputs.push_back(output);
  
  return true;
}

bool LocalResponseNormalizationLayer::
  Connect(const CombinedTensor* input, CombinedTensor* output) {
  // TODO Validate dimensions
  bool valid = true;

  if (!valid) {
    LOGERROR << "Invalid dimensions!";
    return false;
  }
    
  // Save dimensions
  input_width_ = input->data.width();
  input_height_ = input->data.height();
  maps_ = input->data.maps();
  
  return true;
}

void LocalResponseNormalizationLayer::FeedForward() {
  
}

void LocalResponseNormalizationLayer::BackPropagate() {

}


}