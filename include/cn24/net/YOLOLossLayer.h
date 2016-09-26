/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file YOLOLossLayer.h
 * @class YOLOLossLayer
 * @brief This Layer calculates the loss according to the YOLO paper
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_YOLOLOSSLAYER_H
#define CONV_YOLOLOSSLAYER_H

#include <string>
#include <sstream>

#include "Layer.h"
#include "LossFunctionLayer.h"

namespace Conv {
  
class YOLOLossLayer : public Layer, public LossFunctionLayer {
public:
  /**
   * @brief Constructs an YOLOLossLayer.
   */
  YOLOLossLayer(JSON configuration);
  
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
		ss << "YOLO Loss Layer (Weight: " << loss_weight_ << ")";
		return ss.str();
	}

	bool IsDynamicTensorAware() { return true; }
private:
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
  CombinedTensor* third_ = nullptr;

  unsigned int horizontal_cells_ = 0;
  unsigned int vertical_cells_ = 0;
  unsigned int boxes_per_cell_ = 0;
	unsigned int classes_ = 0;

	datum scale_noobj_ = 0.5;
  datum scale_coord_ = 5;

	datum loss_weight_ = 1.0;

	long double current_loss_ = 0;
};

}

#endif
