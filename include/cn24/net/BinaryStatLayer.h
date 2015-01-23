/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file BinaryStatLayer.h
 * \class BinaryStatLayer
 * \brief Class that calculates various binary classification statistics.
 * 
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_BINARYSTATLAYER_H
#define CONV_BINARYSTATLAYER_H

#include <vector>

#include "CombinedTensor.h"
#include "Layer.h"

namespace Conv {

class BinaryStatLayer: public Layer {
public:
  BinaryStatLayer(unsigned int thresholds = 24, const datum min_t = -0.458333,
                  const datum max_t = 0.5);
  void Print(std::string prefix, bool training);
  void Reset();
  
  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs);
  void FeedForward();
  void BackPropagate(); 
  
  inline void SetDisabled(bool disabled = false) {
    disabled_ = disabled;
  }
  
protected:
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
  CombinedTensor* third_ = nullptr;
  
  unsigned int thresholds_ = 0;
  
  datum* threshold_values_ = nullptr;
  datum* true_positives_ = nullptr;
  datum* true_negatives_ = nullptr;
  datum* false_positives_ = nullptr;
  datum* false_negatives_ = nullptr;
  
  bool disabled_ = false;
};

}
  
#endif