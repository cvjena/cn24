/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include "Log.h"
#include "Init.h"

#include "BinaryStatLayer.h"

namespace Conv {

BinaryStatLayer::BinaryStatLayer ( const unsigned int thresholds,
                                   const datum min_t, const datum max_t )
  : thresholds_ ( thresholds ) {
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
}

bool BinaryStatLayer::CreateOutputs ( const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs ) {
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

    /*acc = ( true_positives_[t] + true_negatives_[t] ) /
          ( true_positives_[t] + true_negatives_[t] +
            false_negatives_[t] + false_positives_[t]
          );

    LOGDEBUG << "Accuracy (" << threshold_values_[t] << "): " << acc;*/

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

  ( training ? LOGTRESULT : LOGRESULT )
      << prefix << " F1 : " << f1 * 100.0 << "% (t=" << threshold_values_[tfmax]
      << ")" << LOGRESULTEND;
  ( training ? LOGTRESULT : LOGRESULT )
      << prefix << " ACC: " << acc * 100.0 << "%" << LOGRESULTEND;
  ( training ? LOGTRESULT : LOGRESULT )
      << prefix << " PRE: " << precision * 100.0 << "%" << LOGRESULTEND;
  ( training ? LOGTRESULT : LOGRESULT )
      << prefix << " REC: " << recall * 100.0 << "%" << LOGRESULTEND;
  ( training ? LOGTRESULT : LOGRESULT )
      << prefix << " FPR: " << fpr * 100.0 << "%" << LOGRESULTEND;
  ( training ? LOGTRESULT : LOGRESULT )
      << prefix << " FNR: " << fnr * 100.0 << "%" << LOGRESULTEND;


}



}
