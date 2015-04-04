/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

/**
 * @file UpscaleLayer.h
 * @class UpscaleLayer
 * @brief Layer that scales samples up after MaxPooling downscaled them.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_UPSCALELAYER_H
#define CONV_UPSCALELAYER_H

#include <vector>
#include <string>
#include <sstream>

#include "Tensor.h"
#include "SimpleLayer.h"

namespace Conv {
  
class UpscaleLayer : public SimpleLayer {
public:
  /**
	* @brief Create an UpscaleLayer with the specified borders
	*
	* @param region_width The horizontal border size
	* @param region_height The vertical border size
	*/
  UpscaleLayer(const unsigned int region_width,
	       const unsigned int region_height);
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
  
	inline std::string GetLayerDescription() {
		std::ostringstream ss;
		ss << "Upscale Layer (" << region_width_ << "x" << region_height_ << ")";
		return ss.str();
	}

private:
  // Settings
  unsigned int region_width_ = 0;
  unsigned int region_height_ = 0;
  
  // Feature map dimensions
  unsigned int input_width_ = 0;
  unsigned int input_height_ = 0;
  unsigned int output_width_ = 0;
  unsigned int output_height_ = 0;
  
  unsigned int maps_ = 0;
};
  
}
#endif
