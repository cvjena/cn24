/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cstring>

#include "Config.h"
#include "ConfigParsing.h"
#include "GradientAccumulationLayer.h"


namespace Conv {


GradientAccumulationLayer::GradientAccumulationLayer
  (unsigned int output_count) : Layer (JSON::object()), output_count_(output_count) {
  LOGDEBUG << "Instance created.";
}
  
GradientAccumulationLayer::GradientAccumulationLayer
  (JSON configuration) : Layer(configuration) {
  output_count_ = 0;
	if(configuration.count("outputs") != 1 || !configuration["outputs"].is_number()) {
		FATAL("Invalid configuration (no outputs): " << configuration.dump());
	} else {
		output_count_ = configuration["outputs"];
	}
}

bool GradientAccumulationLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                                 std::vector< CombinedTensor* >& outputs) {
  if(inputs.size() != 1) {
    LOGERROR << "Needs one input!";
    return false;
  }
  
  CombinedTensor* input = inputs[0];
  for(unsigned int i = 0; i < output_count_; i++) {
    CombinedTensor* output = new CombinedTensor(input->data.samples(),
                                                input->data.width(),
                                                input->data.height(),
                                                input->data.maps());
    output->data.Shadow(input->data);
    outputs.push_back(output);
  }

  return true;
}

bool GradientAccumulationLayer::Connect (const std::vector< CombinedTensor* >& inputs,
                           const std::vector< CombinedTensor* >& outputs,
                           const NetStatus* status ) {
  UNREFERENCED_PARAMETER(status);
  if(inputs.size() != 1) {
    LOGERROR << "Needs one input!";
    return false;
  }
  
  CombinedTensor* input = inputs[0];
  
  if(input == nullptr) {
    LOGERROR << "Null pointer supplied";
    return false;
  }
  
  if(outputs.size() != output_count_) {
    LOGERROR << "Wrong number of output nodes!";
    return false;
  }
  
  for(unsigned int i = 0; i < output_count_; i++) {
    if(outputs[i] == nullptr) {
      LOGERROR << "Null pointer supplied";
      return false;
    }
    if(input->data.samples() != outputs[i]->data.samples()) {
      LOGERROR << "Sample count doesn't match!";
      return false;
    }
    if(input->data.elements() != outputs[i]->data.elements()) {
      LOGERROR << "Wrong output dimensions!";
    }
    outputs_.push_back(outputs[i]);
  }
  
  samples_ = input->data.samples();
  elements_per_sample_ = input->data.width() * input->data.height() * input->data.maps();
  input_ = input;
  
  return true;
}

void GradientAccumulationLayer::FeedForward() {
  // Nothing to do here because of the shadowing
}

void GradientAccumulationLayer::BackPropagate() {
  input_->delta.Clear();
#pragma omp parallel for default(shared)
  for(unsigned int s = 0; s < samples_; s++) {
    unsigned int start_element = s * elements_per_sample_;
    for(unsigned int e = 0; e < elements_per_sample_; e++) {
      for(unsigned int i = 0; i < output_count_; i++) {
        input_->delta[start_element + e] += outputs_[i]->delta[start_element + e];
      }
    }
  }
}

}
