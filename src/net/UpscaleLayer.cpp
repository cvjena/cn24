/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <limits>

#include "Log.h"
#include "Init.h"
#include "TensorMath.h"

#include "UpscaleLayer.h"

namespace Conv {

UpscaleLayer::UpscaleLayer ( const unsigned int region_width, const unsigned int region_height )
  : SimpleLayer(JSON::object()), region_width_ ( region_width ), region_height_ ( region_height ) {
  LOGDEBUG << "Instance created: " << region_width_ << "x" << region_height_ <<
           " upscaling.";
}

UpscaleLayer::UpscaleLayer(JSON configuration) : SimpleLayer(configuration) {
  region_width_ = 1;
  region_height_ = 1;

	if(configuration.count("size") != 1 || !configuration["size"].is_array() || configuration["size"].size() != 2) {
		FATAL("Invalid configuration (no size): " << configuration.dump());
	} else {
		region_width_ = configuration["size"][0];
		region_height_ = configuration["size"][0];
	}

  // TODO Validation of actual values
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
  bool valid =
    (input->data.width() * region_height_ == output->data.width())
    && (input->data.height() * region_height_ == output->data.height())
    && (input->data.maps() == output->data.maps())
    && (input->data.samples() == output->data.samples());

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
 TensorMath::UP(input_->data, output_->data, region_width_, region_height_, 1.0f);
}

void UpscaleLayer::BackPropagate() {
  TensorMath::DOWN(output_->delta, input_->delta, region_width_, region_height_, 1.0f);
}

}
