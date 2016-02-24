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

namespace Conv {
  
class HMaxActivationFunction : public SimpleLayer, public LossFunctionLayer {
public:
  HMaxActivationFunction(const datum mu) : mu_(mu) {};
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
  
  CombinedTensor* weights_;
  datum mu_ = 1;
  
  datum sum_of_activations_ = 0;
  datum total_activations_ = 0;
};
}

#endif