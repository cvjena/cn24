/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file LocalResponseNormalizationLayer.h
 * @class LocalResponseNormalizationLayer
 * @brief A local response normaliyation layer using the formula described in
 *   the Caffe Layer Catalogue: http://caffe.berkeleyvision.org/tutorial/layers.html
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_LOCALRESPONSENORMALIZATION_H
#define CONV_LOCALRESPONSENORMALIZATION_H

#include <string>
#include <sstream>

#include "SimpleLayer.h"

namespace Conv {
class LocalResponseNormalizationLayer: public SimpleLayer {
  
public:
  enum NormalizationMethod {
    ACROSS_CHANNELS,
    WITHIN_CHANNELS
  };
  LocalResponseNormalizationLayer(const unsigned int size,
                                  const datum alpha, const datum beta,
                                  const NormalizationMethod normalization_method);

  explicit LocalResponseNormalizationLayer(JSON descriptor);
  
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
  unsigned int size_;
  datum alpha_, beta_;
  NormalizationMethod normalization_method_;
  
  // Feature map dimensions
  unsigned int input_width_ = 0;
  unsigned int input_height_ = 0;
  unsigned int maps_ = 0;
  
  // Buffers
  Tensor region_sums_;
};
}

#endif
