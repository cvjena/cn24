/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file BinaryStatLayer.h
 * @class BinaryStatLayer
 * @brief Class that calculates various binary classification statistics.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_BINARYSTATLAYER_H
#define CONV_BINARYSTATLAYER_H

#include <vector>

#include "CombinedTensor.h"
#include "Layer.h"

namespace Conv {

class BinaryStatLayer: public Layer {
public:
  /**
	* @brief Creates a BinaryStatLayer
	*
	* @param thresholds When computing FMax, the number of binarization thresholds to use
	* @param min_t The minimum binarization threshold
	* @param max_t The maximum binarization threshold
	*/
  BinaryStatLayer(unsigned int thresholds = 24, const datum min_t = -0.458333,
                  const datum max_t = 0.5);

  /**
	* @brief Prints the current statistics
	*
	* @param prefix This is printed before every line ouf output
	* @param training Whether the net is currently training. Affects output color
	*/
  void Print(std::string prefix, bool training);

  /**
	* @brief Reset all counters
	*/
  void Reset();
  
  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs,
                const NetStatus* net );
  void FeedForward();
  void BackPropagate(); 
  
  /**
	* @brief Disables or enables the collection of statistics
	*
	* @param disabled If the collection should be disabled
	*/
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
