/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file MaxPoolingLayer.h
 * @class MaxPoolingLayer
 * @brief Layer that sends the maximum of a specified region to the next.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_MAXPOOLINGLAYER_H
#define CONV_MAXPOOLINGLAYER_H

#include <string>
#include <sstream>

#include "Tensor.h"
#include "SimpleLayer.h"


namespace Conv {
  
class MaxPoolingLayer : public SimpleLayer {
public:
  /**
   * @brief Constructs a max-pooling Layer.
   * 
   * @param region_width Width of the pooling regions
   * @param region_height Height of the pooling regions
   */
  MaxPoolingLayer(const unsigned int region_width,
                  const unsigned int region_height);
  
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

  bool IsOpenCLAware();
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
  
  Tensor maximum_mask_;
  Tensor maximum_ix_;
  Tensor maximum_iy_;
};

}

#endif
