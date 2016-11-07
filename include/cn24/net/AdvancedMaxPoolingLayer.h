/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file AdvancedMaxPoolingLayer.h
 * @class AdvancedMaxPoolingLayer
 * @brief Layer that sends the maximum of a specified region to the next.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_ADVMAXPOOLINGLAYER_H
#define CONV_ADVMAXPOOLINGLAYER_H

#include <string>
#include <sstream>

#include "SimpleLayer.h"


namespace Conv {
  
class AdvancedMaxPoolingLayer : public SimpleLayer {
public:
  /**
   * @brief Constructs a max-pooling Layer.
   * 
   * @param region_width Width of the pooling regions
   * @param region_height Height of the pooling regions
   */
  AdvancedMaxPoolingLayer(const unsigned int region_width,
                  const unsigned int region_height,
                  const unsigned int stride_width,
                  const unsigned int stride_height );
  
  explicit AdvancedMaxPoolingLayer(JSON configuration);
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
  
  inline unsigned int Gain() {
    return gain / (region_width_ * region_height_);
  }
  
	inline std::string GetLayerDescription() {
		std::ostringstream ss;
		ss << "Max-Pooling Layer (" << region_width_ << "x" << region_height_ << ")";
		return ss.str();
	}

  bool IsGPUMemoryAware();
private:
  // Settings
  unsigned int region_width_ = 0;
  unsigned int region_height_ = 0;
  unsigned int stride_width_ = 0;
  unsigned int stride_height_ = 0;
  
  // Feature map dimensions
  unsigned int input_width_ = 0;
  unsigned int input_height_ = 0;
  unsigned int output_width_ = 0;
  unsigned int output_height_ = 0;
  
  unsigned int maps_ = 0;
  
  Tensor maximum_mask_;
};

}

#endif
