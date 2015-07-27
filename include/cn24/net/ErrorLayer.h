/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file ErrorLayer.h
 * @class ErrorLayer
 * @brief This Layer calculates the sum of the quadratic errors from training.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_ERRORLAYER_H
#define CONV_ERRORLAYER_H

#include <string>
#include <sstream>

#include "Layer.h"
#include "LossFunctionLayer.h"

namespace Conv {
  
class ErrorLayer : public Layer, public LossFunctionLayer {
public:
  /**
   * @brief Constructs an ErrorLayer.
   */
  ErrorLayer(const datum loss_weight = 1.0);
  
  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs,
                const NetStatus* net );
  void FeedForward();
  void BackPropagate(); 
  
  // Implementations for LossFunctionLayer
  datum CalculateLossFunction();
  
	std::string GetLayerDescription() {
		std::ostringstream ss;
		ss << "Square Loss Layer (Weight: " << loss_weight_ << ")";
		return ss.str();
	}
private:
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
  CombinedTensor* third_ = nullptr;

	datum loss_weight_ = 1.0;
};

}

#endif
