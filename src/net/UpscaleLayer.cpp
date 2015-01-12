/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <limits>

#include "Log.h"
#include "Init.h"

#include "UpscaleLayer.h"

namespace Conv {

UpscaleLayer::UpscaleLayer ( const unsigned int region_width, const unsigned int region_height )
  :  region_width_ ( region_width ), region_height_ ( region_height ) {
  LOGDEBUG << "Instance created: " << region_width_ << "x" << region_height_ <<
           " upscaling.";
}

bool UpscaleLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs ) {
  // This is a simple layer, only one input
  if ( inputs.size() != 1 ) {
    LOGERROR << "Only one input supported!";
    return false;
  }

  // Save input node pointer
  CombinedTensor* input = inputs[0];

  // Check if input node pointer is null
  if ( input == nullptr ) {
    LOGERROR << "Null pointer input node!";
    return false;
  }

  // Create output
  CombinedTensor* output = new CombinedTensor ( input->data.samples(),
      input->data.width() * region_width_, input->data.height() * region_height_,
      input->data.maps() );

  // Tell network about the output
  outputs.push_back ( output );

  return true;
}

bool UpscaleLayer::Connect ( const CombinedTensor* input,
                             CombinedTensor* output ) {
  // TODO Validate dimensions
  bool valid = true;

  if ( !valid ) {
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

void UpscaleLayer::FeedForward() {
  #pragma omp parallel for default(shared)

  for ( std::size_t sample = 0; sample < input_->data.samples(); sample++ ) {
    for ( unsigned int map = 0; map < maps_; map++ ) {
      for ( unsigned int ox = 0; ox < output_width_; ox++ ) {
        for ( unsigned int oy = 0; oy < output_height_; oy++ ) {
          const unsigned int ix = ox / region_width_;
          const unsigned int iy = oy / region_height_;
          const datum ival = *input_->data.data_ptr_const ( ix, iy, map, sample );
          // Feed forward
          *output_->data.data_ptr ( ox, oy, map, sample ) = ival;
        }
      }
    }
  }
}

void UpscaleLayer::BackPropagate() {
  #pragma omp parallel for default(shared)

  for ( std::size_t sample = 0; sample < input_->data.samples(); sample++ ) {
    for ( unsigned int map = 0; map < maps_; map++ ) {
      for ( unsigned int ix = 0; ix < input_width_; ix++ ) {
        for ( unsigned int iy = 0; iy < input_height_; iy++ ) {
          const unsigned int ox = ix * region_width_;
          const unsigned int oy = iy * region_height_;
          datum sum = 0;

          for ( unsigned int ry = 0; ry < region_height_; ry++ ) {
            for ( unsigned int rx = 0; rx < region_width_; rx++ ) {
              sum += *output_->delta.data_ptr_const ( ox + rx, oy +ry, map, sample );
            }
          }

          *input_->delta.data_ptr ( ix,iy,map,sample ) = sum / (datum)(region_width_ * region_height_);
        }
      }
    }
  }

  return;

}

}
