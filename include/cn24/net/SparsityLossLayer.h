/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file SparsityLossLayer.h
 * @class SparsityLossLayer
 * @brief This Layer calculates the l1/l2 sparsity loss
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_SPARSITYLOSSLAYER_H
#define CONV_SPARSITYLOSSLAYER_H

#include <string>
#include <sstream>

#include "Layer.h"
#include "../util/JSONParsing.h"
#include "LossFunctionLayer.h"


namespace Conv {
  
class SparsityLossLayer : public Layer, public LossFunctionLayer {
public:
  /**
   * @brief Constructs a SparsityLossLayer.
   */
  SparsityLossLayer(JSON configuration);
  
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
		ss << "Sparsity Loss Layer (Weight: " << loss_weight_ << ")";
		return ss.str();
	}
private:

	unsigned int stat_id_ = 0;
	StatDescriptor stat_descriptor_;
  CombinedTensor* first_ = nullptr;

	const datum zero_threshold = 0.00005;
  datum loss_weight_ = 1.0;
};

}

#endif
