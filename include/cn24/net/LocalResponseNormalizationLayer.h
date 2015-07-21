/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file GradientAccumulationLayer.h
 * @class GradientAccumulationLayer
 * @brief Accumulates the gradients at the outputs
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_LOCALRESPONSENORMALIZATION_H
#define CONV_LOCALRESPONSENORMALIZATION_H

#include <string>
#include <sstream>

#include "SimpleLayer.h"
#include "NetGraph.h"
#include "CombinedTensor.h"

namespace Conv {
class LocalResponseNormalizationLayer: public SimpleLayer {
  
public:
  enum NormalizationMethod {
    ACROSS_CHANNELS,
    WITHIN_CHANNELS
  };
  LocalResponseNormalizationLayer(const unsigned int rx, const unsigned int ry,
                                  const datum alpha, const datum beta,
                                  const NormalizationMethod normalization_method);
  
  // SimpleLayer implementations
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  bool Connect(const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
  
  std::string GetLayerDescription() { return "Local Response Normalization Layer"; }
  void CreateBufferDescriptors(std::vector< NetGraphBuffer >& buffers) {
      NetGraphBuffer buffer;
      buffer.description = "Output";
      buffers.push_back(buffer);
  }
  
private:
  unsigned int rx_, ry_;
  datum alpha_, beta_;
  NormalizationMethod normalization_method_;
  
  // Feature map dimensions
  unsigned int input_width_ = 0;
  unsigned int input_height_ = 0;
  unsigned int maps_ = 0;
};
}

#endif