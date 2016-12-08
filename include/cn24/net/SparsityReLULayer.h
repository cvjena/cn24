/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_SPARITYRELULAYER_H
#define CONV_SPARITYRELULAYER_H

#include <string>
#include <sstream>

#include "SimpleLayer.h"
#include "LossFunctionLayer.h"
#include "../util/StatAggregator.h"

namespace Conv {
  
class SparsityReLULayer : public SimpleLayer, public LossFunctionLayer {
public:
  SparsityReLULayer(const datum lambda, const datum kl_loss_weight);
  explicit SparsityReLULayer(JSON configuration);
  
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
    ss << "Sparsity ReLU (lambda: " << lambda_ << ")";
    return ss.str();
  }
  
  // Implementations for Layer
  bool IsNotGradientSafe() {
    return kl_loss_weight_ > 0.0;
  }
  
  
  CombinedTensor* weights_;
  datum lambda_ = 1;
  datum kl_loss_weight_ = 0;
  datum other_loss_weight_ = 0;
  datum alpha_ = 0.5;
};
}

#endif
