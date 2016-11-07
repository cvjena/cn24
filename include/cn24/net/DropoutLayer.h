/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file DropoutLayer.h
 * @class DropoutLayer
 * @brief Layer that applies dropout during FF and BP passes
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_DROPOUTLAYER_H
#define CONV_DROPOUTLAYER_H

#include <string>
#include <random>

#include "SimpleLayer.h"

namespace Conv {

class DropoutLayer : public SimpleLayer {
public:
  explicit DropoutLayer(JSON configuration);
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  virtual void FeedForward();
  virtual void BackPropagate();

  virtual std::string GetLayerDescription() { return "Dropout Layer"; }

  virtual bool IsGPUMemoryAware() { return true; }
private:
  std::mt19937 rand_;
  datum dropout_fraction_;
  Tensor dropout_mask_;
  datum scale_;
  unsigned int seed_;
};

}

#endif
