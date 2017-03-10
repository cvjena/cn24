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
#include "TensorMath.h"

namespace Conv {

LocalResponseNormalizationLayer::
  LocalResponseNormalizationLayer(const unsigned int size,
    const datum alpha, const datum beta,
    const LocalResponseNormalizationLayer::NormalizationMethod normalization_method) :
  SimpleLayer(JSON::object()),
  size_(size), alpha_(alpha), beta_(beta), normalization_method_(normalization_method) {
  LOGDEBUG << "Instance created, size: " << size_ << ", alpha: " << alpha_ 
  << ", beta: " << beta_ << ", method: " << ((normalization_method_ == ACROSS_CHANNELS) ? "across" : "within");

}

LocalResponseNormalizationLayer::
  LocalResponseNormalizationLayer(JSON descriptor) :
  SimpleLayer(descriptor) {
  JSON_TRY_INT(size_, descriptor, "size", 5);
  JSON_TRY_DATUM(alpha_, descriptor, "alpha", 0.0001);
  JSON_TRY_DATUM(beta_, descriptor, "beta", 0.75);
  normalization_method_ = ACROSS_CHANNELS;

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
  UNREFERENCED_PARAMETER(output);
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
  
  
  // Resize region sum buffer
  region_sums_.Resize(input->data.samples(), input_width_, input_height_, maps_);
  
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
            int region_size = 0;
            for(int iy = (((int)y-sub) > 0 ? (int)y-sub : 0); (iy < (int)input_height_) && (iy <= ((int)y+add)); iy++) {
              for(int ix = (((int)x-sub) > 0 ? (int)x-sub : 0); (ix < (int)input_width_) && (ix <= ((int)x+add)); ix++) {
                const datum input_value = (*input_->data.data_ptr_const((const size_t)ix,(const size_t)iy,map,sample));
                region_sum += input_value * input_value;
                region_size++;
              }
            }
            
            (*region_sums_.data_ptr(x,y,map,sample)) = region_sum;
            datum divisor = (datum)pow(1.0 + ((alpha_/((datum)region_size))*region_sum), beta_);
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
            for(int imap = (((int)map-sub) > 0 ? (int)map-sub : 0); (imap < (int)maps_) && (imap <= ((int)map+add)); imap++) {
              const datum input_value = (*input_->data.data_ptr_const(x,y,(const size_t)imap,sample));
              region_sum += input_value * input_value;
              region_size++;
            }
            
            (*region_sums_.data_ptr(x,y,map,sample)) = region_sum;
            datum divisor = (datum)pow(1.0 + ((alpha_/((datum)region_size))*region_sum), beta_);
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
  TensorMath::SETSAMPLE(input_->delta, -1, 0);
  
  
  const int sub = (size_-1)/2;
  const int add = (size_)/2;
  #pragma omp parallel for default(shared)
  for(unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    if(normalization_method_ == WITHIN_CHANNELS) {
      for(unsigned int map = 0; map < maps_; map++) {
        for(unsigned int y = 0; y < input_height_; y++) {
          for(unsigned int x = 0; x < input_height_; x++) {
            datum region_sum = (*region_sums_.data_ptr_const(x,y,map,sample));
            unsigned int region_size = 0;
            for(int iy = (((int)y-sub) > 0 ? (int)y-sub : 0); (iy < (int)input_height_) && (iy <= ((int)y+add)); iy++) {
              for(int ix = (((int)x-sub) > 0 ? (int)x-sub : 0); (ix < (int)input_width_) && (ix <= ((int)x+add)); ix++) {
                region_size++;
              }
            }
            
            datum divisor = (datum)pow(1.0 + ((alpha_/((datum)region_size))*region_sum), beta_);
            datum divisor2 = divisor * divisor;
            const datum xi = *input_->data.data_ptr_const(x,y,map,sample);
            const datum dxi = *output_->delta.data_ptr_const(x,y,map,sample);
            
            for(int iy = (((int)y-sub) > 0 ? (int)y-sub : 0); (iy < (int)input_height_) && (iy <= ((int)y+add)); iy++) {
              for(int ix = (((int)x-sub) > 0 ? (int)x-sub : 0); (ix < (int)input_width_) && (ix <= ((int)x+add)); ix++) {
                const datum xj = *input_->data.data_ptr_const((const size_t)ix,(const size_t)iy,map,sample);
                (*input_->delta.data_ptr((const size_t)ix,(const size_t)iy,map,sample)) -= (dxi * xi * (datum)2.0 * beta_ * (alpha_ / (datum)region_size) * xj *
                  (datum)pow((datum)1.0 + (alpha_ / (datum)region_size) * region_sum, beta_ - (datum)1.0)) / divisor2;
              }
            }
            (*input_->delta.data_ptr(x,y,map,sample)) += dxi / divisor;
          }
        }
      }
    } else if(normalization_method_ == ACROSS_CHANNELS) {
      for(unsigned int map = 0; map < maps_; map++) {
        for(unsigned int y = 0; y < input_height_; y++) {
          for(unsigned int x = 0; x < input_height_; x++) {
            datum region_sum = (*region_sums_.data_ptr_const(x,y,map,sample));
            unsigned int region_size = 0;
            for(int imap = (((int)map-sub) > 0 ? (int)map-sub : 0); (imap < (int)maps_) && (imap <= ((int)map+add)); imap++) {
              region_size++;
            }
            
            datum divisor = (datum)pow(1.0 + ((alpha_/((datum)region_size))*region_sum), beta_);
            datum divisor2 = divisor * divisor;
            const datum xi = *input_->data.data_ptr_const(x,y,map,sample);
            const datum dxi = *output_->delta.data_ptr_const(x,y,map,sample);
            
            for(int imap = (((int)map-sub) > 0 ? (int)map-sub : 0); (imap < (int)maps_) && (imap <= ((int)map+add)); imap++) {
                const datum xj = *input_->data.data_ptr_const(x,y,(const size_t)imap,sample);
                (*input_->delta.data_ptr(x,y,(const size_t)imap,sample)) -= (dxi * xi * (datum)2.0 * beta_ * (alpha_ / (datum)region_size) * xj *
                  (datum)pow((datum)1.0 + (alpha_ / (datum)region_size) * region_sum, beta_ - (datum)1.0)) / divisor2;
            }
            (*input_->delta.data_ptr(x,y,map,sample)) += dxi / divisor;
          }
        }
      }
    } else {
      FATAL("Unknown normalization method");
    }
  }
}

}
