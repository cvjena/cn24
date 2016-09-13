/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <string>
#include <iomanip>
#include <sstream>
#include "../util/StatAggregator.h"

#include "ConfusionMatrixLayer.h"

namespace Conv {
ConfusionMatrixLayer::ConfusionMatrixLayer ( ClassManager* class_manager ) :
  Layer(JSON::object()), class_manager_(class_manager)
{

  LOGDEBUG << "Instance created, " << class_manager_->GetClassCount() << " classes for now.";

  UpdateClassCount();

  // Initialize stat descriptors
  stat_orr_ = new StatDescriptor;
  stat_arr_ = new StatDescriptor;
  stat_iou_ = new StatDescriptor;
  
  stat_orr_->description = "Overall Recognition Rate";
  stat_orr_->unit = "%";
  stat_orr_->nullable = true;
  stat_orr_->init_function = [this] (Stat& stat) { stat.is_null = true; stat.value = 0; Reset();};
  stat_orr_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_orr_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat {
    UNREFERENCED_PARAMETER(hc_stats);
    return stat;
  };
  
  stat_arr_->description = "Average Recognition Rate";
  stat_arr_->unit = "%";
  stat_arr_->nullable = true;
  stat_arr_->init_function = [] (Stat& stat) { stat.is_null = true; stat.value = 0; };
  stat_arr_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_arr_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat {
    UNREFERENCED_PARAMETER(hc_stats);
    return stat;
  };
  
  stat_iou_->description = "Average Intersection over Union";
  stat_iou_->unit = "%";
  stat_iou_->nullable = true;
  stat_iou_->init_function = [] (Stat& stat) { stat.is_null = true; stat.value = 0; };
  stat_iou_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_iou_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat {
    UNREFERENCED_PARAMETER(hc_stats);
    return stat;
  };
  
  // Register with StatAggregator
  System::stat_aggregator->RegisterStat(stat_orr_);
  System::stat_aggregator->RegisterStat(stat_arr_);
  System::stat_aggregator->RegisterStat(stat_iou_);
}
  
void ConfusionMatrixLayer::UpdateAll() {
  // Don't call Update(...) when there are no samples to keep the null property of the value
  if (total_ < 1.0)
    return;
  
  long double orr = 0, arr = 0, iou = 0;
  
  // Calculate metrics...
  
  // Overall recognition rate
  orr = 100.0L * right_ / total_;

  // Average recognition rate
  long double ccount = 0;
  long double sum = 0;

  for(ClassManager::const_iterator it = class_manager_->begin(); it != class_manager_->end(); it++) {
    unsigned int c = it->second.id;
    if ( per_class_[c] > 0 ) {
      sum += matrix_[ ( c * (last_seen_max_id_ + 1) ) + c] / per_class_[c];
      ccount += 1.0L;
    }
  }

  arr = 100.0L * sum / ccount;

  // Intersection over union
  long double IU_sum = 0;
  for(ClassManager::const_iterator itt = class_manager_->begin(); itt != class_manager_->end(); itt++) {
    unsigned int t = itt->second.id;
    // Calculate IU measure for class T
    long double unionn = 0;
    for(ClassManager::const_iterator it = class_manager_->begin(); it != class_manager_->end(); it++) {
      unsigned int c = it->second.id;
      if(c!=t) {
        unionn += matrix_[ ( t * (last_seen_max_id_ + 1) ) + c];
        unionn += matrix_[ ( c * (last_seen_max_id_ + 1) ) + t];
      }
    }
    unionn += matrix_[ ( t * (last_seen_max_id_ + 1)) + t];
    long double IU = (unionn > 0.0) ? (matrix_[ ( t * (last_seen_max_id_ + 1)) + t] / unionn) : 0.0;
    IU_sum += IU;
  }

  iou = 100.0L * IU_sum / (long double)(last_seen_max_id_ + 1);
  
  // Submit metrics to StatAggregator
  System::stat_aggregator->Update(stat_orr_->stat_id, (double)orr);
  System::stat_aggregator->Update(stat_arr_->stat_id, (double)arr);
  System::stat_aggregator->Update(stat_iou_->stat_id, (double)iou);
}

void ConfusionMatrixLayer::UpdateClassCount() {
  if(last_seen_max_id_ != class_manager_->GetMaxClassId() || matrix_ == nullptr) {
    unsigned int max_id = class_manager_->GetMaxClassId();

    if(matrix_ != nullptr)
      delete[] matrix_;
    if(per_class_ != nullptr)
      delete[] per_class_;

    matrix_ = new long double[(max_id + 1) * (max_id)];
    per_class_ = new long double[(max_id + 1)];

    last_seen_max_id_ = max_id;

    Reset();
  }
}

bool ConfusionMatrixLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs ) {
  UNREFERENCED_PARAMETER(outputs);
  // Validate input node count
  if ( inputs.size() != 3 ) {
    LOGERROR << "Need exactly 3 inputs to calculate confusion matrix!";
    return false;
  }

  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];
  CombinedTensor* third = inputs[2];

  // Check for null pointers
  if ( first == nullptr || second == nullptr || third == nullptr ) {
    LOGERROR << "Null pointer node supplied";
    return false;
  }

  if ( first->data.elements() != second->data.elements() ) {
    LOGERROR << "Inputs need the same number of elements!";
    return false;
  }

  if ( first->data.samples() != third->data.samples() ) {
    LOGERROR << "Inputs need the same number of samples!";
    return false;
  }

  // Needs no outputs
  return true;
}

bool ConfusionMatrixLayer::Connect
( const std::vector< CombinedTensor* >& inputs,
  const std::vector< CombinedTensor* >& outputs,
  const NetStatus* net ) {
  UNREFERENCED_PARAMETER(net);
  // Needs exactly three inputs to calculate the stat
  if ( inputs.size() != 3 )
    return false;

  // Also, the two inputs have to have the same number of samples and elements!
  // We ignore the shape for now...
  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];
  CombinedTensor* third = inputs[2];
  bool valid = first != nullptr && second != nullptr &&
               first->data.samples() == second->data.samples() &&
               first->data.samples() == third->data.samples() &&
               outputs.size() == 0;

  if ( valid ) {
    first_ = first;
    second_ = second;
    third_ = third;

    UpdateClassCount();
  }

  return valid;
}

void ConfusionMatrixLayer::FeedForward() {
  if ( disabled_ )
    return;

  UpdateClassCount();

  for ( unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    for ( unsigned int y = 0; y < first_->data.height(); y++ ) {
      for ( unsigned int x = 0; x < first_->data.width(); x++ ) {
        unsigned int first_class = first_->data.PixelMaximum ( x,y,sample );
        unsigned int second_class = second_->data.PixelMaximum ( x,y,sample );
        const long double weight = *third_->data.data_ptr_const ( x,y,0,sample );
        matrix_[ ( first_class * (last_seen_max_id_ + 1) ) + second_class] += weight;
        per_class_[second_class] += weight;
        total_ += weight;

        if ( first_class == second_class )
          right_ += weight;
      }
    }
  }
}

void ConfusionMatrixLayer::BackPropagate() {

}


void ConfusionMatrixLayer::Reset() {
  for ( unsigned int c = 0; c < ( (last_seen_max_id_ + 1) * (last_seen_max_id_ + 1) ); c++ ) {
    matrix_[c] = 0;
  }

  for ( unsigned int c = 0; c < (last_seen_max_id_ + 1); c++ ) {
    per_class_[c] = 0;
  }

  total_ = 0;
  right_ = 0;
}

void ConfusionMatrixLayer::Print ( std::string prefix, bool training ) {
  // Print confusion matrix
  std::stringstream caption;
  caption << std::setw ( 12 ) << "vCLS  ACT>";

  unsigned int classes_ = last_seen_max_id_ + 1;

  for ( unsigned int c = 0; c < classes_; c++ ) {
    caption << std::setw ( 12 ) << class_manager_->GetClassInfoById(c).first;
  }

  (training?LOGTRESULT:LOGRESULT) << caption.str() << LOGRESULTEND;
  caption.str ( "" );


  for ( unsigned int t = 0; t < classes_; t++ ) {
    caption << std::setw ( 12 ) << class_manager_->GetClassInfoById(t).first;

    for ( unsigned int c = 0; c < classes_; c++ ) {
      long double result = matrix_[ ( t * classes_ ) + c];
      caption << std::setw ( 12 ) << static_cast<long> ( result );
    }

    (training?LOGTRESULT:LOGRESULT) << caption.str() << LOGRESULTEND;
    caption.str ( "" );
  }
  
  // Average recognition rate
  for ( unsigned int c = 0; c < classes_; c++ ) {
    if ( per_class_[c] > 0 ) {
      long double classrr = matrix_[ ( c * classes_ ) + c] / per_class_[c];
      caption << std::setw(12) << class_manager_->GetClassInfoById(c).first;
      caption << " RR: ";
      caption << std::setw(12) << classrr * 100.0L;
      caption << "%";
      (training?LOGTRESULT:LOGRESULT) << prefix << caption.str() << LOGRESULTEND;
      caption.str ( "" );
    }
  }


  // Print IOU
  long double IU_sum = 0;
  
  for(unsigned int t = 0; t < classes_; t++) {
    // Calculate IU measure for class T
    long double unionn = 0;
    for(unsigned int c = 0; c < classes_; c++) {
      if(c!=t) {
        unionn += matrix_[ ( t * classes_ ) + c];
        unionn += matrix_[ ( c * classes_ ) + t];
      }
    }
    unionn += matrix_[ ( t * classes_) + t];
    long double IU = (unionn > 0.0) ? (matrix_[ ( t * classes_) + t] / unionn) : 0.0;
    IU_sum += IU;
    caption << std::setw(12) << class_manager_->GetClassInfoById(t).first;
    caption << " IU: ";
    caption << std::setw(12) << IU * 100.0L;
    caption << "%";
    (training?LOGTRESULT:LOGRESULT) << prefix << caption.str() << LOGRESULTEND;
    caption.str ( "" );
  }

  (training?LOGTRESULT:LOGRESULT) << prefix << " Average intersection over union          : "
            << 100.0L * IU_sum / (long double)classes_ << "%" << LOGRESULTEND;
}

ConfusionMatrixLayer::~ConfusionMatrixLayer() {
  if ( matrix_ != nullptr )
    delete[] matrix_;
}




}
