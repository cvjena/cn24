/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include "Log.h"
#include "Init.h"
#include "StatAggregator.h"

#include "BinaryStatLayer.h"

namespace Conv {

BinaryStatLayer::BinaryStatLayer ( const unsigned int thresholds,
                                   const datum min_t, const datum max_t )
  : Layer(JSON::object()), thresholds_ ( thresholds ) {
  if ( thresholds < 1 ) {
    FATAL ( "Binary classification needs at least one threshold value!" );
  }

  LOGDEBUG << "Instance created. Using " << thresholds_ << " thresholds from " <<
           min_t << " to " << max_t;
  threshold_values_ = new datum[thresholds_];
  true_positives_ = new datum[thresholds_];
  false_positives_ = new datum[thresholds_];
  true_negatives_ = new datum[thresholds_];
  false_negatives_ = new datum[thresholds_];

  if ( thresholds_ == 1 ) {
    threshold_values_[0] = ( min_t + max_t ) / 2.0;
  } else {
    datum interval = ( max_t - min_t ) / ( datum ) ( thresholds_ - 1 );

    for ( unsigned int t = 0; t < thresholds_; t++ ) {
      threshold_values_[t] = min_t + interval * ( datum ) t;
    }
  }

  Reset();

  // Initialize stat descriptors
  stat_fpr_ = new StatDescriptor;
  stat_fnr_ = new StatDescriptor;
  stat_pre_ = new StatDescriptor;
  stat_rec_ = new StatDescriptor;
  stat_acc_ = new StatDescriptor;
  stat_f1_ = new StatDescriptor;

  stat_fpr_->description = "False Positive Rate";
  stat_fpr_->unit = "%";
  stat_fpr_->nullable = true;
  stat_fpr_->init_function = [this] (Stat& stat) { stat.is_null = true; stat.value = 0; Reset(); };
  stat_fpr_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_fpr_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  stat_fnr_->description = "False Negative Rate";
  stat_fnr_->unit = "%";
  stat_fnr_->nullable = true;
  stat_fnr_->init_function = [] (Stat& stat) { stat.is_null = true; stat.value = 0; };
  stat_fnr_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_fnr_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  stat_pre_->description = "Precision";
  stat_pre_->unit = "%";
  stat_pre_->nullable = true;
  stat_pre_->init_function = [] (Stat& stat) { stat.is_null = true; stat.value = 0; };
  stat_pre_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_pre_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  stat_rec_->description = "Recall";
  stat_rec_->unit = "%";
  stat_rec_->nullable = true;
  stat_rec_->init_function = [] (Stat& stat) { stat.is_null = true; stat.value = 0; };
  stat_rec_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_rec_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  stat_acc_->description = "Accuracy";
  stat_acc_->unit = "%";
  stat_acc_->nullable = true;
  stat_acc_->init_function = [] (Stat& stat) { stat.is_null = true; stat.value = 0; };
  stat_acc_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_acc_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  stat_f1_->description = "F1 Value";
  stat_f1_->unit = "%";
  stat_f1_->nullable = true;
  stat_f1_->init_function = [] (Stat& stat) { stat.is_null = true; stat.value = 0; };
  stat_f1_->update_function = [] (Stat& stat, double user_value) { stat.is_null = false; stat.value = user_value; };
  stat_f1_->output_function = [] (HardcodedStats& hc_stats, Stat& stat) -> Stat { UNREFERENCED_PARAMETER(hc_stats); return stat; };

  // Register stats
  System::stat_aggregator->RegisterStat(stat_fpr_);
  System::stat_aggregator->RegisterStat(stat_fnr_);
  System::stat_aggregator->RegisterStat(stat_pre_);
  System::stat_aggregator->RegisterStat(stat_rec_);
  System::stat_aggregator->RegisterStat(stat_acc_);
  System::stat_aggregator->RegisterStat(stat_f1_);
}

void BinaryStatLayer::UpdateAll() {
  // Calculate metrics
  datum fmax = -2;
  unsigned int tfmax = -1;

  for ( unsigned int t = 0; t < thresholds_; t++ ) {
    datum precision = -1;
    datum recall = -1;
    datum f1 = -1;

    if ( ( true_positives_[t] + false_positives_[t] ) > 0 )
      precision = ( true_positives_[t] ) /
                  ( true_positives_[t] + false_positives_[t] );

    if ( ( true_positives_[t] + false_negatives_[t] ) > 0 )
      recall = ( true_positives_[t] ) /
               ( true_positives_[t] + false_negatives_[t] );

    if ( precision >= 0 && recall >= 0 ) {
      f1 = 2 * precision * recall / ( precision + recall );
    }

    if ( f1 > fmax ) {
      fmax = f1;
      tfmax = t;
    }
  }

  datum fpr = -1;
  datum fnr = -1;
  datum precision = -1;
  datum recall = -1;
  datum f1 = -1;
  datum acc = -1;

  if ( ( true_positives_[tfmax] + false_positives_[tfmax] ) > 0 )
    precision = ( true_positives_[tfmax] ) /
                ( true_positives_[tfmax] + false_positives_[tfmax] );

  if ( ( true_positives_[tfmax] + false_negatives_[tfmax] ) > 0 )
    recall = ( true_positives_[tfmax] ) /
             ( true_positives_[tfmax] + false_negatives_[tfmax] );

  if ( ( false_positives_[tfmax] + true_negatives_[tfmax] ) > 0 )
    fpr = ( false_positives_[tfmax] ) /
          ( false_positives_[tfmax] + true_negatives_[tfmax] );

  if ( ( true_positives_[tfmax] + false_negatives_[tfmax] ) > 0 )
    fnr = ( false_negatives_[tfmax] ) /
          ( true_positives_[tfmax] + false_negatives_[tfmax] );

  if ( precision >= 0 && recall >= 0 )
    f1 = 2 * precision * recall / ( precision + recall );
  
  acc = ( true_positives_[tfmax] + true_negatives_[tfmax] ) /
          ( true_positives_[tfmax] + true_negatives_[tfmax] +
            false_negatives_[tfmax] + false_positives_[tfmax]
          );

  // Update stats
  if(fpr >= 0) System::stat_aggregator->Update(stat_fpr_->stat_id, 100.0 * fpr);
  if(fnr >= 0) System::stat_aggregator->Update(stat_fnr_->stat_id, 100.0 * fnr);
  if(precision >= 0) System::stat_aggregator->Update(stat_pre_->stat_id, 100.0 * precision);
  if(recall >= 0) System::stat_aggregator->Update(stat_rec_->stat_id, 100.0 * recall);
  if(acc >= 0) System::stat_aggregator->Update(stat_acc_->stat_id, 100.0 * acc);
  if(f1 >= 0) System::stat_aggregator->Update(stat_f1_->stat_id, 100.0 * f1);
}

bool BinaryStatLayer::CreateOutputs ( const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs ) {
  UNREFERENCED_PARAMETER(outputs);
  // Validate input node count
  if ( inputs.size() != 3 ) {
    LOGERROR << "Need exactly 3 inputs to calculate binary stat!";
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

  if ( first->data.samples() != second->data.samples() ) {
    LOGERROR << "Inputs need the same number of samples!";
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

bool BinaryStatLayer::Connect ( const std::vector< CombinedTensor* >& inputs, const std::vector< CombinedTensor* >& outputs, const NetStatus* net ) {
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
               first->data.elements() == second->data.elements() &&
               first->data.samples() == third->data.samples() &&
               outputs.size() == 0;

  if ( valid ) {
    first_ = first;
    second_ = second;
    third_ = third;
  }

  return valid;
}

void BinaryStatLayer::FeedForward() {
  if ( disabled_ )
    return;

  #pragma omp parallel for default(shared)
  for ( unsigned int t = 0; t < thresholds_; t++ ) {
    for ( std::size_t s = 0; s < first_->data.elements(); s++ ) {
      const bool sign = first_->data ( s ) > threshold_values_[t];
      const bool expected_sign = second_->data ( s ) > 0; //threshold_values_[t];
      const datum weight = third_->data ( s );

      if ( sign && expected_sign )
        true_positives_[t] += weight;

      if ( sign && !expected_sign )
        false_positives_[t] += weight;

      if ( !sign && expected_sign )
        false_negatives_[t] += weight;

      if ( !sign && !expected_sign )
        true_negatives_[t] += weight;
    }
  }
}

void BinaryStatLayer::BackPropagate() {

}

void BinaryStatLayer::Reset() {

  for ( unsigned int t = 0; t < thresholds_; t++ ) {
    true_negatives_[t] = 0;
    true_positives_[t] = 0;
    false_negatives_[t] = 0;
    false_positives_[t] = 0;
  }
}

void BinaryStatLayer::Print ( std::string prefix, bool training ) {
  UNREFERENCED_PARAMETER(prefix);
  UNREFERENCED_PARAMETER(training);
  // Now deprecated
}



}
