/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file ConcatLayer.h
 * \class ConcatLayer
 * \brief Concatenates the inputs (used to add non-convolvable information).
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CONCATLAYER_H
#define CONV_CONCATLAYER_H

#include "Layer.h"
#include "CombinedTensor.h"

namespace Conv {

class ConcatLayer: public Layer {
public:
  ConcatLayer();

  // Layer implementations
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs);
  void FeedForward();
  void BackPropagate();

private:
  CombinedTensor* input_a_ = nullptr;
  CombinedTensor* input_b_ = nullptr;
  CombinedTensor* output_ = nullptr;
  
  unsigned int width_a_ = 0;
  unsigned int width_b_ = 0;
  unsigned int samples_ = 0;
};

}
#endif
