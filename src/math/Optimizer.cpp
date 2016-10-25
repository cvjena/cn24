/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Optimizer.h"

namespace Conv {
Optimizer::Optimizer(JSON configuration) : configuration_(configuration) {
  // Don't do anything, lazy initialization happens in Step(3)
}

void Reset() {
  ResetInner();
}

void Optimizer::Step(const std::vector<CombinedTensor *> &parameters, unsigned int iteration) {
  // If the optimizer uses buffers, check if parameter count has changed
  if(GetRequiredBufferCount() > 0) {
    std::size_t parameter_count = 0;
    for(CombinedTensor* const ctensor : parameters) {
      parameter_count += ctensor->data.elements();
    }

    // Check if the actual buffer count is okay


    // If it has, resize all the buffers
    if(buffers_[0]->elements() != parameter_count) {
      for(Tensor* buffer : buffers_) {
        buffer->Resize(parameter_count);
      }
      LOGDEBUG << "Resized optimization buffer(s) to " << parameter_count << " elements each.";
    }
  }

  // Call inner step
  Step(buffers_, parameters, iteration);
}

}
