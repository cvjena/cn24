/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <limits>
#include <cmath>

#include "Log.h"
#include "LocalResponseNormalizationLayer.h"

namespace Conv {

LocalResponseNormalizationLayer::
  LocalResponseNormalizationLayer(const unsigned int size,
    const datum alpha, const datum beta,
    const LocalResponseNormalizationLayer::NormalizationMethod normalization_method) :
    size_(size), alpha_(alpha), beta_(beta), normalization_method_(normalization_method) {
  LOGDEBUG << "Instance created, size: " << size_ << ", alpha: " << alpha_ 
  << ", beta: " << beta_ << ", method: " << ((normalization_method_ == ACROSS_CHANNELS) ? "across" : "within");

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
  const int sub = (size_-1)/2;
  const int add = (size_)/2;
  #pragma omp parallel for default(shared)
  for(unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    if(normalization_method_ == WITHIN_CHANNELS) {
      for(unsigned int map = 0; map < maps_; map++) {
        for(unsigned int y = 0; y < input_height_; y++) {
          for(unsigned int x = 0; x < input_height_; x++) {
            datum region_sum = 0;
            unsigned int region_size = 0;
            for(unsigned int iy = (((int)y-sub) > 0 ? (int)y-sub : 0); (iy < input_height_) && (iy <= ((int)y+add)); iy++) {
              for(unsigned int ix = (((int)x-sub) > 0 ? (int)x-sub : 0); (ix < input_width_) && (ix <= ((int)x+add)); ix++) {
                const datum input_value = (*input_->data.data_ptr_const(ix,iy,map,sample));
                region_sum += input_value * input_value;
                region_size++;
              }
            }
            
            datum divisor = pow(1.0 + ((alpha_/((datum)region_size))*region_sum), beta_);
            (*output_->data.data_ptr(x,y,map,sample)) = (*input_->data.data_ptr_const(x,y,map,sample)) / divisor;
          }
        }
      }
    } else if(normalization_method_ == ACROSS_CHANNELS) {
      for(unsigned int map = 0; map < maps_; map++) {
        for(unsigned int y = 0; y < input_height_; y++) {
          for(unsigned int x = 0; x < input_height_; x++) {
            datum region_sum = 0;
            unsigned int region_size = 0;
            for(unsigned int imap = (((int)map-sub) > 0 ? (int)map-sub : 0); (imap < maps_) && (imap <= ((int)map+add)); imap++) {
              const datum input_value = (*input_->data.data_ptr_const(x,y,imap,sample));
              region_sum += input_value * input_value;
              region_size++;
            }
            
            datum divisor = pow(1.0 + ((alpha_/((datum)region_size))*region_sum), beta_);
            (*output_->data.data_ptr(x,y,map,sample)) = (*input_->data.data_ptr_const(x,y,map,sample)) / divisor;
          }
        }
      }
    } else {
      FATAL("Unknown normalization method");
    }
  }
}

void LocalResponseNormalizationLayer::BackPropagate() {
  FATAL("Backward pass missing, use only for prediction!");
}


}
