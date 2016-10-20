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
#include "../util/ClassManager.h"

namespace Conv {


class DetectionStatLayer: public Layer, public StatLayer {
public:
  struct Detection {
  public:
    datum confidence = 0;
    datum tp = 0;
    datum fp = 0;
  };

  /**
	* @brief Creates a DetectionStatLayer
	*
  * @param class_manager Global ClassManager instance
	*/
  explicit DetectionStatLayer(ClassManager* class_manager);

  void UpdateClassCount();
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

  bool IsDynamicTensorAware() { return true; }

	std::string GetLayerDescription() { return "Detection Statistic Layer"; }
protected:
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
  CombinedTensor* third_ = nullptr;

  ClassManager* class_manager_ = nullptr;
  unsigned int last_seen_max_id_ = 0;

  std::vector<Detection>* detections_ = nullptr;
  unsigned int* positive_samples_ = nullptr;

  datum objectness_tp_ = 0;
  datum objectness_fp_ = 0;
  datum objectness_positives_ = 0;
  
  bool disabled_ = false;

  StatDescriptor* stat_obj_pre_ = nullptr;
  StatDescriptor* stat_obj_rec_ = nullptr;
  StatDescriptor* stat_map_ = nullptr;

  StatDescriptor* stat_obj_f1_ = nullptr;
  StatDescriptor* stat_mf1_ = nullptr;
};

}
  
#endif
