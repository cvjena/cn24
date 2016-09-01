/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file DetectionStatLayer.h
 * @class DetectionStatLayer
 * @brief Class that calculates various binary classification statistics.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_DETECTIONSTATLAYER_H
#define CONV_DETECTIONSTATLAYER_H

#include <vector>

#include "Layer.h"
#include "StatLayer.h"	
#include "../util/StatAggregator.h"

namespace Conv {

class DetectionStatLayer: public Layer, public StatLayer {
public:
  /**
	* @brief Creates a DetectionStatLayer
	*
	* @param thresholds When computing FMax, the number of binarization thresholds to use
	* @param min_t The minimum binarization threshold
	* @param max_t The maximum binarization threshold
	*/
  DetectionStatLayer(const unsigned int classes, unsigned int thresholds = 24, const datum min_t = -0.458333,
                  const datum max_t = 0.5);
  
  void UpdateAll();

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
  
	std::string GetLayerDescription() { return "Detection Statistic Layer"; }
protected:
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;

  unsigned int thresholds_ = 0;
  unsigned int classes_ = 0;
  
  datum* threshold_values_ = nullptr;
  datum* true_positives_ = nullptr;
  datum* true_negatives_ = nullptr;
  datum* false_positives_ = nullptr;
  datum* false_negatives_ = nullptr;
  
  bool disabled_ = false;

  StatDescriptor* stat_fpr_ = nullptr;
  StatDescriptor* stat_fnr_ = nullptr;
  StatDescriptor* stat_pre_ = nullptr;
  StatDescriptor* stat_rec_ = nullptr;
  StatDescriptor* stat_acc_ = nullptr;
  StatDescriptor* stat_f1_ = nullptr;
};

}
  
#endif
