/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include "Log.h"
#include "TensorMath.h"

#include "InputDownSamplingLayer.h"

namespace Conv {

InputDownSamplingLayer::InputDownSamplingLayer (const unsigned int region_width,
                                  const unsigned int region_height) :
  SimpleLayer(JSON::object()),
  region_width_ (region_width), region_height_ (region_height) {
  LOGDEBUG << "Instance created: " << region_width_ << "x" << region_height_ <<
           " pooling.";
}

bool InputDownSamplingLayer::CreateOutputs (
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
  if ( (input->data.width() % region_width_) != 0 ||
       (input->data.height() % region_height_) != 0) {
    LOGERROR << "Input dimensions not divisible by region dimensions!";
    return false;
  }

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      input->data.width() / region_width_, input->data.height() / region_height_,
      input->data.maps());

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool InputDownSamplingLayer::Connect (const CombinedTensor* input,
                               CombinedTensor* output) {
  // TODO Validate dimensions
  bool valid = true;

  if (!valid) {
    LOGERROR << "Invalid dimensions!";
    return false;
  }

  // Save dimensions
  input_width_ = input->data.width();
  input_height_ = input->data.height();
  output_width_ = output->data.width();
  output_height_ = output->data.height();

  maps_ = input->data.maps();

  return true;
}

void InputDownSamplingLayer::FeedForward() {
  TensorMath::DOWN(input_->data, output_->data, region_width_, region_height_, 1.0f / ((datum)region_width_ * (datum)region_height_));
}

void InputDownSamplingLayer::BackPropagate() {
  if(backprop_enabled_) {
    FATAL("This is a pre-processing layer that does not support backpropagation!");
  }
}


bool InputDownSamplingLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL_MAX
  return true;
#else
  return false;
#endif
}

}
