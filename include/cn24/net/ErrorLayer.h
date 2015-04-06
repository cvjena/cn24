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

#include "Config.h"
#include "CombinedTensor.h"
#include "Layer.h"
#include "LossFunctionLayer.h"

namespace Conv {
  
class ErrorLayer : public Layer, public LossFunctionLayer {
public:
  /**
   * @brief Constructs an ErrorLayer.
   */
  ErrorLayer();
  
  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs,
                const Net* net );
  void FeedForward();
  void BackPropagate(); 
  
  // Implementations for LossFunctionLayer
  datum CalculateLossFunction();
  
	std::string GetLayerDescription() { return "Square Loss Layer";  }
private:
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
  CombinedTensor* third_ = nullptr;
};

}

#endif
