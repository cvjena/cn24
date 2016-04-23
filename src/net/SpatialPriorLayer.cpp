/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "SpatialPriorLayer.h"
namespace Conv {
  SpatialPriorLayer::SpatialPriorLayer() : SimpleLayer(JSON::object()) {
  LOGDEBUG << "Instance created.";
}

bool SpatialPriorLayer::CreateOutputs ( const std::vector< CombinedTensor* >& inputs,
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
      input->data.width(),
      input->data.height(),
      input->data.maps() + 2 );
  // Tell network about the output
  outputs.push_back ( output );

  return true;
}

bool SpatialPriorLayer::Connect ( const CombinedTensor* input,
                                  CombinedTensor* output ) {
  if ( input == nullptr || output == nullptr ) {
    FATAL ( "Null pointer given!" );
    return false;
  }

  if ( input->data.maps() != output->data.maps() - 2 ||
       input->data.width() != output->data.width() ||
       input->data.height() != output->data.height() ||
       input->data.samples() != output->data.samples() ) {
    FATAL ( "Dimensions don't match!" );
    return false;
  }

  return true;
}

void SpatialPriorLayer::FeedForward() {
  output_->data.Clear ( 1.0 );
  
  #pragma omp parallel for default(shared)
  for ( unsigned int sample = 0; sample < input_->data.samples(); sample++ ) {
    for ( unsigned int map = 2; map < input_->data.maps() + 2; map++ ) {
      for ( unsigned int y = 0; y < input_->data.height(); y++ ) {
        Tensor::CopyMap ( input_->data, sample, map-2,
                          output_->data, sample, map );
      }
    }

    for ( unsigned int y = 0; y < input_->data.height(); y++ ) {
      for ( unsigned int x = 0; x < input_->data.width(); x++ ) {
        // Copy x helper
        *output_->data.data_ptr ( x,y,0,sample ) = ( ( datum ) x ) / ( ( datum ) input_->data.width() );
        // Copy y helper
        *output_->data.data_ptr ( x,y,1,sample ) = ( ( datum ) y ) / ( ( datum ) input_->data.height() );
      }
    }
  }
}

void SpatialPriorLayer::BackPropagate() {
  #pragma omp parallel for default(shared)
  for ( unsigned int sample = 0; sample < input_->data.samples(); sample++ ) {
    for ( unsigned int map = 2; map < input_->data.maps() + 2; map++ ) {
      for ( unsigned int y = 0; y < input_->data.height(); y++ ) {
        Tensor::CopyMap ( output_->delta, sample, map,
                          input_->delta, sample, map - 2 );
      }
    }
  }
}


}
