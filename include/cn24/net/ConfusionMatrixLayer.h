/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file ConfusionMatrixLayer.h
 * @class ConfusionMatrixLayer
 * @brief Represents a layer that calculates a confusion matrix
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CONFUSIONMATRIXLAYER_H
#define CONV_CONFUSIONMATRIXLAYER_H

#include <string>
#include <vector>

#include "Layer.h"
#include "StatLayer.h"

#include "../util/StatAggregator.h"
#include "../util/ClassManager.h"

namespace Conv {

class ConfusionMatrixLayer: public Layer, public StatLayer {
public:
  explicit ConfusionMatrixLayer(ClassManager* class_manager);

  void UpdateClassCount();
  void UpdateAll();
  /**
	* @brief Prints the current statistics
	*
	* @param prefix This is printed before every line of output
	* @param training Whether the net is currently training. Affects output color
	*/
  void Print (std::string prefix, bool training);

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
  inline void SetDisabled (bool disabled = false) {
    disabled_ = disabled;
  }

	std::string GetLayerDescription() { return "Confusion Matrix Layer"; }
  ~ConfusionMatrixLayer();
private:
  bool disabled_ = false;

  ClassManager* class_manager_ = nullptr;
  unsigned int last_seen_max_id_ = 0;

  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
  CombinedTensor* third_ = nullptr;
  
  long double* matrix_ = nullptr;
  long double total_ = 0;
  long double right_ = 0;
  long double* per_class_ = nullptr;
  
  StatDescriptor* stat_orr_ = nullptr;
  StatDescriptor* stat_arr_ = nullptr;
  StatDescriptor* stat_iou_ = nullptr;
  
};

}

#endif
