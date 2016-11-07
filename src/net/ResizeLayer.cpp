/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include "Log.h"

#include <cstring>

#include "ResizeLayer.h"
#include "Init.h"

#include "ConfigParsing.h"

#ifdef BUILD_OPENMP
#include <omp.h>
#endif

namespace Conv {

ResizeLayer::ResizeLayer (const unsigned int borderx,
			  const unsigned int bordery) :
  SimpleLayer(JSON::object()),
  borderx_(borderx), bordery_(bordery) {
  LOGDEBUG << "Instance created, border size: (" << borderx << ", "
  << bordery << ")";
}

ResizeLayer::ResizeLayer (JSON configuration)
: SimpleLayer(configuration) {
	if(configuration.count("border") != 1 || !configuration["border"].is_array() || configuration["border"].size() != 2) {
		FATAL("Invalid configuration (no size): " << configuration.dump());
	} else {
		borderx_ = configuration["border"][0];
		bordery_ = configuration["border"][0];
	}
}
  
bool ResizeLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
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

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
					       input->data.width() + borderx_,
					       input->data.height() + bordery_,
					       input->data.maps() );
  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool ResizeLayer::Connect (const CombinedTensor* input,
                            CombinedTensor* output) {
  if (input == nullptr || output == nullptr) {
    FATAL ("Null pointer given!");
    return false;
  }

  if (input->data.maps() != output->data.maps() ||
      input->data.width() + borderx_ != output->data.width() ||
      input->data.height() + bordery_ != output->data.height() ||
      input->data.samples() != output->data.samples()) {
    FATAL ("Dimensions don't match!");
    return false;
  }

  return true;
}

void ResizeLayer::FeedForward() {
#ifdef BUILD_OPENCL
  output_->data.MoveToCPU(true);
  input_->data.MoveToCPU();
#endif
  
  output_->data.Clear(0.0);
#pragma omp parallel for default(shared)
  for(unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for(unsigned int map = 0; map < input_->data.maps(); map++) {
      for(unsigned int y = 0; y < input_->data.height(); y++) {
	const datum* const source =
	  input_->data.data_ptr_const(0, y, map, sample);
	datum* const target =
	  output_->data.data_ptr(borderx_ / 2, y + (bordery_ / 2), map, sample);
	  
	std::memcpy(target, source, sizeof(datum) * input_->data.width());
      }
    }
  }
}

void ResizeLayer::BackPropagate() {
  // Nothing to do here
  if(backprop_enabled_) {
#ifdef BUILD_OPENCL
    input_->delta.MoveToCPU(true);
    output_->delta.MoveToCPU();
#endif
    
    input_->delta.Clear(0.0);
#pragma omp parallel for default(shared)
    for(unsigned int sample = 0; sample < input_->data.samples(); sample++) {
      for(unsigned int map = 0; map < input_->data.maps(); map++) {
        for(unsigned int y = 0; y < input_->data.height(); y++) {
  	datum* const target =
  	  input_->delta.data_ptr(0, y, map, sample);
  	const datum* const source =
  	  output_->delta.data_ptr_const(borderx_ / 2, y + (bordery_ / 2), map, sample);
  	  
  	std::memcpy(target, source, sizeof(datum) * input_->delta.width());
        }
      }
    }
      
  }
}

bool ResizeLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL
  return true; 
#else
  return false; 
#endif
}

}
