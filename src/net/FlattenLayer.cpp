/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include "Log.h"

#include "FlattenLayer.h"

namespace Conv {

FlattenLayer::FlattenLayer() {
  LOGDEBUG << "Instance created.";
}

bool FlattenLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
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
  CombinedTensor* output = new CombinedTensor (0, 0, 0, 0);
  output->data.Shadow (input->data);
  output->delta.Shadow (input->delta);

  std::size_t new_width = input->data.width() * input->data.height() *
                          input->data.maps();

  std::size_t samples = input->data.samples();

  output->data.Reshape (samples, new_width);
  output->delta.Reshape (samples, new_width);

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool FlattenLayer::Connect (const CombinedTensor* input,
                            CombinedTensor* output) {
  if (input == nullptr || output == nullptr) {
    FATAL ("Null pointer given!");
    return false;
  }

  if (input->data.elements() != output->data.elements() ||
      input->data.samples() != output->data.samples()) {
    FATAL ("Dimensions don't match!");
    return false;
  }

  if (input->data.data_ptr() != output->data.data_ptr() ||
      input->delta.data_ptr() != output->delta.data_ptr()) {
    FATAL ("Output Tensor doesn't shadow input Tensor!");
    return false;
  }

  return true;
}

void FlattenLayer::FeedForward() {
  // Nothing to do here
}

void FlattenLayer::BackPropagate() {
  // Nothing to do here
}

}
