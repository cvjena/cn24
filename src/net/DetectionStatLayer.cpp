/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <algorithm>
#include <vector>

#include "Log.h"
#include "Init.h"
#include "StatAggregator.h"

#include "DetectionStatLayer.h"

namespace Conv {

DetectionStatLayer::DetectionStatLayer ( const unsigned int classes )
  : Layer(JSON::object()), classes_(classes) {

  LOGDEBUG << "Instance created.";

  Reset();

  // Initialize stat descriptors
  stat_tpr_ = new StatDescriptor;
  stat_fpr_ = new StatDescriptor;
  stat_map_ = new StatDescriptor;

  stat_tpr_->description = "True Positive Rate";
  stat_tpr_->unit = "%";
  stat_tpr_->nullable = true;
  stat_tpr_->init_function = [this] (Stat& stat) { stat.is_null = true; stat.value = 0; Reset(); };
  stat_tpr_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_tpr_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  stat_fpr_->description = "False Positive Rate";
  stat_fpr_->unit = "%";
  stat_fpr_->nullable = true;
  stat_fpr_->init_function = [this] (Stat& stat) { stat.is_null = true; stat.value = 0; Reset(); };
  stat_fpr_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_fpr_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  stat_map_->description = "mAP";
  stat_map_->unit = "%";
  stat_map_->nullable = true;
  stat_map_->init_function = [] (Stat& stat) { stat.is_null = true; stat.value = 0; };
  stat_map_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_map_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  // Register stats
  System::stat_aggregator->RegisterStat(stat_tpr_);
  System::stat_aggregator->RegisterStat(stat_fpr_);
  System::stat_aggregator->RegisterStat(stat_map_);
}

void DetectionStatLayer::UpdateAll() {
  // Sort vector
  std::sort(detections_.begin(), detections_.end(), [](Detection& d1, Detection& d2) { return d1.confidence < d2.confidence; });

  // Calculate metrics

  // Update stats
  /*if(tpr >= 0) System::stat_aggregator->Update(stat_fpr_->stat_id, 100.0 * fpr);
  if(fpr >= 0) System::stat_aggregator->Update(stat_fpr_->stat_id, 100.0 * fpr);
  if(fnr >= 0) System::stat_aggregator->Update(stat_fnr_->stat_id, 100.0 * fnr);
  if(precision >= 0) System::stat_aggregator->Update(stat_map_->stat_id, 100.0 * precision);
  if(recall >= 0) System::stat_aggregator->Update(stat_rec_->stat_id, 100.0 * recall);
  if(acc >= 0) System::stat_aggregator->Update(stat_acc_->stat_id, 100.0 * acc);
  if(f1 >= 0) System::stat_aggregator->Update(stat_f1_->stat_id, 100.0 * f1);*/
}

bool DetectionStatLayer::CreateOutputs ( const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs ) {
  UNREFERENCED_PARAMETER(outputs);
  // Validate input node count
  if ( inputs.size() < 2 ) {
    LOGERROR << "Need at least 2 inputs to calculate detection stat!";
    return false;
  }

  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];

  // Check for null pointers
  if ( first == nullptr || second == nullptr) {
    LOGERROR << "Null pointer node supplied";
    return false;
  }

  if ( first->data.samples() != second->data.samples() ) {
    LOGERROR << "Inputs need the same number of samples!";
    return false;
  }

  if ( first->metadata == nullptr || second->metadata == nullptr) {
    LOGERROR << "Both inputs need metadata for bounding boxes!";
    return false;
  }

  // Needs no outputs
  return true;
}

bool DetectionStatLayer::Connect ( const std::vector< CombinedTensor* >& inputs, const std::vector< CombinedTensor* >& outputs, const NetStatus* net ) {
  UNREFERENCED_PARAMETER(net);
  // Needs exactly three inputs to calculate the stat
  if ( inputs.size() < 2 )
    return false;

  // Also, the two inputs have to have the same number of samples and elements!
  // We ignore the shape for now...
  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];
  CombinedTensor* third = inputs[2];
  bool valid = first != nullptr && second != nullptr &&
               first->data.samples() == second->data.samples() &&
               first->metadata != nullptr &&
               second->metadata != nullptr &&
               outputs.size() == 0;

  if ( valid ) {
    first_ = first;
    second_ = second;
  }

  return valid;
}

void DetectionStatLayer::FeedForward() {
  if ( disabled_ )
    return;
}

void DetectionStatLayer::BackPropagate() {

}

void DetectionStatLayer::Reset() {
  detections_.clear();
}

void DetectionStatLayer::Print ( std::string prefix, bool training ) {
  UNREFERENCED_PARAMETER(prefix);
  UNREFERENCED_PARAMETER(training);
  // Now deprecated
}



}
