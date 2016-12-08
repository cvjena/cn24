/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
/**
 * @file HMaxActivationFunction.h
 * @class HMaxActivationFunction
 * @brief This layer introduces a non-linearity (activation function)
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_HMAXACTIVATIONFUNCTION_H
#define CONV_HMAXACTIVATIONFUNCTION_H

#include <string>
#include <sstream>

#include "SimpleLayer.h"
#include "LossFunctionLayer.h"
#include "../util/StatAggregator.h"

namespace Conv {
  
class HMaxActivationFunction : public SimpleLayer, public LossFunctionLayer {
public:
  HMaxActivationFunction(const datum mu, const datum loss_weight);
  explicit HMaxActivationFunction(JSON configuration);
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
  
  // Implementations for LossFunctionLayer
  datum CalculateLossFunction();
  
  std::string GetLayerDescription() {
    std::stringstream ss;
    ss << "H-Max Activation Function (mu: " << mu_ << ")";
    return ss.str();
  }
  
  // Implementations for Layer
  bool IsNotGradientSafe() {
    return loss_weight_ > 0.0;
  }
  
  
  CombinedTensor* weights_;
  datum mu_ = 1;
  datum loss_weight_ = 0;
  
  datum sum_of_activations_ = 0;
  datum total_activations_ = 0;
  
  StatDescriptor desc_a, desc_b, desc_s;
  
  static int stat_id_a;
  static int stat_id_b;
  static int stat_id_s;

  datum sum_x = 0.0;
  datum sum_x_sq = 0.0;
};
}

#endif
