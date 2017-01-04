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

Optimizer::~Optimizer() {
  // RAII takes care of our tensors
}

void Optimizer::Reset() {
  // LOGINFO << "Resetting optimizer...";
  iterations_since_reset_ = 0;
  ResetInner();
  for(std::vector<Tensor>& buffer_set : buffers_)
    for(Tensor& buffer : buffer_set)
      buffer.Clear();
}

void Optimizer::ResetInner() {
  // Do nothing here
}

void Optimizer::Step(const std::vector<CombinedTensor *> &parameters, unsigned int iteration) {
  bool needs_reset = false;
  // If the optimizer uses buffers, check if parameter count has changed
  if(GetRequiredBufferCount() > 0) {
    // Check if the actual buffer count is okay
    if(buffers_.size() != parameters.size()) {
      needs_reset = true;
      for(std::vector<Tensor>& buffer_set : buffers_) {
        buffer_set.clear();
      }
      buffers_.clear();

      // After this loop, buffers_.size() is equal to parameters.size()
      for(unsigned int p = 0; p < parameters.size(); p++) {
        buffers_.push_back({});
      }
    }

    // Make sure that every buffers_[.] matches required buffer count
    for(unsigned int p = 0; p < parameters.size(); p++) {
      if(buffers_[p].size() != GetRequiredBufferCount()) {
        needs_reset = true;
        buffers_[p].clear();

        // Create new ones
        buffers_[p].resize(GetRequiredBufferCount());
        for(unsigned int b = 0; b < GetRequiredBufferCount(); b++) {
          buffers_[p][b].Resize(parameters[p]->data.elements());
        }
      } else {
        // Check if inner parameter count matches
        for(unsigned int b = 0; b < GetRequiredBufferCount(); b++) {
          if(buffers_[p][b].elements() != parameters[p]->data.elements()) {
            needs_reset = true;
            buffers_[p][b].Resize(parameters[p]->data.elements());
          }
        }
      }
    }
 }

  if(needs_reset)
    Reset();

#ifdef BUILD_OPENCL
  // Need to get the parameter buffers off of the GPU
  if(!IsGPUMemoryAware()) {
    for (unsigned int p = 0; p < parameters.size(); p++) {
      parameters[p]->data.MoveToCPU();
      parameters[p]->delta.MoveToCPU();
    }
  }
#endif

#pragma omp parallel for default(shared)
  for(unsigned int p = 0; p < parameters.size(); p++) {
    // Call inner step
    Step(buffers_[p], parameters[p], iteration);
  }

  iterations_since_reset_++;
}

}
