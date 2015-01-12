/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <cmath>

#ifdef BUILD_OPENMP
#include <omp.h>
#endif

#include "Log.h"
#include "CombinedTensor.h"

#include "CrossEntropyErrorLayer.h"

namespace Conv {

CrossEntropyErrorLayer::CrossEntropyErrorLayer(const unsigned int classes)
: classes_(classes) {
  LOGDEBUG << "Instance created, " << classes << " classes.";
#ifdef ERROR_LAYER_IGNORE_WEIGHTS
  LOGINFO << "Weights are being ignored!";
#endif
}

bool CrossEntropyErrorLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                                std::vector< CombinedTensor* >& outputs) {
  // Validate input node count
  if (inputs.size() != 3) {
    LOGERROR << "Need exactly 3 inputs to calculate loss function!";
    return false;
  }

  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];
  CombinedTensor* third = inputs[2];

  // Check for null pointers
  if (first == nullptr || second == nullptr || third == nullptr) {
    LOGERROR << "Null pointer node supplied";
    return false;
  }

  if (first->data.samples() != second->data.samples()) {
    LOGERROR << "Inputs need the same number of samples!";
    return false;
  }

  if (first->data.elements() != (classes_ * first->data.samples())) {
    LOGERROR << "Inputs needs to match the class count";
    return false;
  }

  if (first->data.samples() != third->data.samples()) {
    LOGERROR << "Inputs need the same number of samples!";
    return false;
  }
  
  // Needs no outputs
  return true;
}

bool CrossEntropyErrorLayer::Connect (const std::vector< CombinedTensor* >& inputs,
                          const std::vector< CombinedTensor* >& outputs) {
  // Needs exactly three inputs to calculate the difference
  if (inputs.size() != 3) 
    return false;

  // Also, the two inputs have to have the same number of samples and elements!
  // We ignore the shape for now...
  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];
  CombinedTensor* third = inputs[2];
  bool valid = first != nullptr && second != nullptr &&
               first->data.samples() == second->data.samples() &&
               first->data.elements() == (classes_ * first->data.samples()) &&
               first->data.samples() == third->data.samples() &&
               outputs.size() == 0;

  if (valid) {
    first_ = first;
    second_ = second;
    third_ = third;
  }

  return valid;
}

void CrossEntropyErrorLayer::FeedForward() {
  // We write the deltas at this point, because
  // CalculateLossFunction() is called before BackPropagate().
  // We don't precalculate the loss because it is not calculated for every
  // batch.
#pragma omp parallel for default(shared)
  for (std::size_t i = 0; i < first_->data.samples(); i++) {
    const unsigned int second = *((duint*)second_->data.data_ptr_const(i));
#ifdef ERROR_LAYER_IGNORE_WEIGHTS
    const datum weight = 1.0;
#else
    const datum weight = third_->data.data_ptr_const() [i];
#endif
    for(unsigned int c = 0; c < classes_; c++) {
      const datum diff = (datum)(*(first_->data.data_ptr_const(c, 0, 0, i)))
      - (datum)(c == second); // ? 1.0 : 0.0);
      *(first_->delta.data_ptr(c, 0, 0, i)) = diff * weight;
    }
  }

}

void CrossEntropyErrorLayer::BackPropagate() {
  // The deltas are already written in to the input CombinedTensors, so
  // there is nothing to do now.
}

datum CrossEntropyErrorLayer::CalculateLossFunction() {
  datum error = 0;

  for (std::size_t i = 0; i < first_->data.samples(); i++) {
    const unsigned int second = *((duint*)second_->data.data_ptr_const(i));
#ifdef ERROR_LAYER_IGNORE_WEIGHTS
    const datum weight = 1.0;
#else
    const datum weight = third_->data.data_ptr_const() [i];
#endif
    for(unsigned int c = 0; c < classes_; c++) {
      const datum diff = (datum)(log(*(first_->data.data_ptr_const(c, 0, 0, i))))
      * (datum)(c == second); // ? 1.0 : 0.0);
      error += diff * weight;
    }
  }

  return -error / first_->data.samples();
}


}
