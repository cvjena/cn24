/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <algorithm>
#include <vector>
#include <cn24/util/BoundingBox.h>

#include "Log.h"
#include "Init.h"
#include "StatAggregator.h"

#include "DetectionStatLayer.h"

namespace Conv {

DetectionStatLayer::DetectionStatLayer (ClassManager* class_manager)
  : Layer(JSON::object()), class_manager_(class_manager) {

  LOGDEBUG << "Instance created, " << class_manager_->GetClassCount() << " classes right now.";


  UpdateClassCount();

  // Initialize stat descriptors
  stat_tpr_ = new StatDescriptor;
  stat_fpr_ = new StatDescriptor;
  stat_map_ = new StatDescriptor;
  stat_rec_ = new StatDescriptor;

  stat_tpr_->description = "True Positive Rate";
  stat_tpr_->unit = "%";
  stat_tpr_->nullable = true;
  stat_tpr_->init_function = [this] (Stat& stat) { stat.is_null = true; stat.value = 0; Reset(); };
  stat_tpr_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_tpr_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  stat_rec_->description = "Recall";
  stat_rec_->unit = "%";
  stat_rec_->nullable = true;
  stat_rec_->init_function = [this] (Stat& stat) { stat.is_null = true; stat.value = 0; Reset(); };
  stat_rec_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_rec_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

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
  System::stat_aggregator->RegisterStat(stat_rec_);
}

void DetectionStatLayer::UpdateClassCount() {
  if(last_seen_max_id_ != class_manager_->GetMaxClassId() || detections_ == nullptr) {
    unsigned int max_id = class_manager_->GetMaxClassId();

    if(detections_ != nullptr)
      delete[] detections_;
    if(positive_samples_ != nullptr)
      delete[] positive_samples_;

    detections_ = new std::vector<Detection>[max_id + 1];
    positive_samples_ = new unsigned int[max_id + 1];
    last_seen_max_id_ = max_id;

    Reset();
  }
}

void DetectionStatLayer::UpdateAll() {
  // Global metrics
  datum global_tpr = 0;
  datum global_fpr = 0;
  datum global_rec = 0;
  datum global_ap = 0;

  datum sampled_classes = 0;
  for(ClassManager::const_iterator it = class_manager_->begin(); it != class_manager_->end(); it++) {
    unsigned int c = it->second.id;
    // Skip empty classes
    if(detections_[c].size() == 0)
      continue;

    // Sort vector
    std::sort(detections_[c].begin(), detections_[c].end(),
              [](Detection &d1, Detection &d2) { return d1.confidence < d2.confidence; });

    // For AP calculation
    datum fp_sum = 0;
    datum tp_sum = 0;
    std::vector<datum> detection_recall;
    std::vector<datum> detection_precision;

    // Initialize AP calculation
    detection_precision.push_back((datum)0);
    detection_recall.push_back((datum)0);

    // Calculate TPR and FPR
    for(unsigned int d = 0; d < detections_[c].size(); d++) {
      tp_sum += detections_[c][d].tp;
      fp_sum += detections_[c][d].fp;

      // Save results for AP calculation
      detection_precision.push_back(tp_sum / (fp_sum + tp_sum));
      detection_recall.push_back(tp_sum / (datum)positive_samples_[c]);
    }

    datum rec = (tp_sum > 0) ? tp_sum / (datum)positive_samples_[c] : 0;
    datum tpr = tp_sum / (datum)(detections_[c].size());
    datum fpr = fp_sum / (datum)(detections_[c].size());

    global_tpr += tpr;
    global_fpr += fpr;
    global_rec += rec;
    sampled_classes += (datum)1.0;

    // Calculate AP
    detection_precision.push_back((datum)0);
    detection_recall.push_back((datum)1);

    for(int i = (int)detection_precision.size() - 2; i >= 0; --i) {
      detection_precision[i] = std::max(detection_precision[i], detection_precision[i+1]);
    }
    std::vector<int> different_indices;
    for(int i = 1; i < detection_recall.size(); i++) {
      if(detection_recall[i] != detection_recall[i-1])
        different_indices.push_back(i);
    }

    datum ap = 0;
    for(int i = 0; i < different_indices.size(); i++) {
      ap += (detection_recall[different_indices[i]] - detection_recall[different_indices[i] - 1]) * detection_precision[different_indices[i]];
    }
    LOGDEBUG << "AP class " << it->first << ": " << ap * 100.0;
    global_ap += ap;
  }
  if(sampled_classes > 0) System::stat_aggregator->Update(stat_tpr_->stat_id, 100.0 * global_tpr / sampled_classes);
  if(sampled_classes > 0) System::stat_aggregator->Update(stat_fpr_->stat_id, 100.0 * global_fpr / sampled_classes);
  if(sampled_classes > 0) System::stat_aggregator->Update(stat_rec_->stat_id, 100.0 * global_rec / sampled_classes);
  if(sampled_classes > 0) System::stat_aggregator->Update(stat_map_->stat_id, 100.0 * global_ap / sampled_classes);

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

  UpdateClassCount();

  for (unsigned int sample = 0; sample < first_->data.samples(); sample++) {
    std::vector<BoundingBox> *sample_detected_boxes = (std::vector<BoundingBox> *) first_->metadata[sample];
    std::vector<BoundingBox> *sample_truth_boxes = (std::vector<BoundingBox> *) second_->metadata[sample];

    for(ClassManager::const_iterator it = class_manager_->begin(); it != class_manager_->end(); it++) {
      unsigned int c = it->second.id;
      // Loop over all detected boxes
      for(unsigned int b = 0; b < sample_detected_boxes->size(); b++) {
        if((*sample_detected_boxes)[b].c != c)
          continue;

        Detection detection;
        detection.confidence = (*sample_detected_boxes)[b].c;

        // Assign ground truth box
        datum maximum_overlap = (datum) -1;
        int best_truth_box = -1;

        for (unsigned int t = 0; t < sample_truth_boxes->size(); t++) {
          if((*sample_truth_boxes)[t].c != c)
            continue;

          // Compute overlap
          datum overlap = (*sample_detected_boxes)[b].IntersectionOverUnion(&((*sample_truth_boxes)[t]));
          if(overlap > maximum_overlap) {
            best_truth_box = t;
            maximum_overlap = overlap;
          }
        }

        if(maximum_overlap > 0.5 && best_truth_box >= 0) { // Second part shouldn't be necessary, but who knows
          if((*sample_truth_boxes)[best_truth_box].flag2) // Difficult box, ignore completely
            continue;

          if(!(*sample_truth_boxes)[best_truth_box].flag1) {
            detection.tp = 1.0;
            (*sample_truth_boxes)[best_truth_box].flag1 = true;
          } else {
            // Double detection -> false positive
            detection.fp = 1.0;
          }
        } else {
          // No box found or box too small
          detection.fp = 1.0;
        }

        detections_[c].push_back(detection);
      }
    }

    // Reset box flags
    for (unsigned int t = 0; t < sample_truth_boxes->size(); t++) {
      (*sample_truth_boxes)[t].flag1 = false;

      // Count positive samples (ignore difficult boxes)
      if(!(*sample_truth_boxes)[t].flag2)
        positive_samples_[(*sample_truth_boxes)[t].c]++;
    }
  }


}

void DetectionStatLayer::BackPropagate() {

}

void DetectionStatLayer::Reset() {
  for (unsigned int c = 0; c < last_seen_max_id_; c++) {
    detections_[c].clear();
    positive_samples_[c] = 0;
  }
}

void DetectionStatLayer::Print ( std::string prefix, bool training ) {
  UNREFERENCED_PARAMETER(prefix);
  UNREFERENCED_PARAMETER(training);
  // Now deprecated
}



}
