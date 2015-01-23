/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file FullyConnectedLayer.h
 * \class FullyConnectedLayer
 * \brief Represents a fully connected layer of neurons.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_FULLYCONNECTEDLAYER_H
#define CONV_FULLYCONNECTEDLAYER_H

#include <vector>
#include <random>

#include "CombinedTensor.h"
#include "SimpleLayer.h"
#include "SupportsDropoutLayer.h"

namespace Conv {

class FullyConnectedLayer : public SimpleLayer, public SupportsDropoutLayer {
public:
  /**
   * \brief Constructs a FullyConnectedLayer.
   *
   * \param neurons Number of neurons in the Layer
   * \param seed Seed for random number generation
   */
  explicit FullyConnectedLayer (const unsigned int neurons, const int seed = 0);

  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  void FeedForward();
  void BackPropagate();
  
  void OnLayerConnect (Layer* next_layer);
  
  // Implementations for SimpleLayer
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  
  inline unsigned int Gain() {
    return input_units_;
  }
  
#ifdef BUILD_OPENCL
  bool IsOpenCLAware() { return true; }
#endif
private:
  unsigned int neurons_ = 0;
  unsigned int input_units_ = 0;
  
  CombinedTensor* weights_ = nullptr;
  CombinedTensor* bias_ = nullptr;
  
  datum* ones_ = nullptr;
  
  std::mt19937 rand_;

};

}

#endif
