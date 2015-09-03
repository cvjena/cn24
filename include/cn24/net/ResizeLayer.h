/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file ResizeLayer.h
 * @class ResizeLayer
 * Layer that adds a border around an image so that it has its original
 * size after convolutions.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_RESIZELAYER_H
#define CONV_RESIZELAYER_H

#include <string>
#include <sstream>

#include "SimpleLayer.h"

namespace Conv {

class ResizeLayer : public SimpleLayer {
public:  
  /**
   * @brief Constructs a ResizeLayer with the specified border size.
   * @param borderx Size of the complete horizontal border
   * @param bordery Size of the complete vertical border
   */
  ResizeLayer(const unsigned int borderx, const unsigned int bordery);
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
  bool IsOpenCLAware();

	inline std::string GetLayerDescription() {
		std::ostringstream ss;
		ss << "Resize Layer (" << borderx_ << "x" << bordery_ << ")";
		return ss.str();
	}
private:
  unsigned int borderx_;
  unsigned int bordery_;
};

}

#endif
