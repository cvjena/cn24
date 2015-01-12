/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <cmath>
#include "DropoutLayer.h"

namespace Conv {

void DropoutLayer::FeedForward() {
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    if ((dist_ (generator_) < dropout_frac_) && do_dropout_) {
      input_->data[element] = 0;
      input_->delta[element] = -1;
    } else {
      input_->delta[element] = 1;
    }
    
    const datum input_data = input_->data (element);
    //if(do_dropout_)
      output_->data [element] = input_data;
    //else
    //  output_->data [element] = input_data * (1.0 - dropout_frac_);
  }
}

void DropoutLayer::BackPropagate() {
  #pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_delta = input_->delta(element);
    const datum output_delta = output_->delta(element);
    if(input_delta == 1) {
      input_->delta[element] = output_delta;
    } else {
      input_->delta[element] = 0;
    }
  }
}


}
