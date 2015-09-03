/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <string>
#include <iomanip>
#include <sstream>
#include "ConfusionMatrixLayer.h"

namespace Conv {
ConfusionMatrixLayer::ConfusionMatrixLayer (
  std::vector<std::string> names, const unsigned int classes ) :
  classes_ ( classes ), names_ ( names ) {
  LOGDEBUG << "Instance created, " << classes << " classes.";
  for(unsigned int n = 0; n < names_.size(); n++) {
    if(names_[n].length() > 11) {
      std::string original = names_[n];
      names_[n] = original.substr(0,8) + "...";
    }
  }
}

bool ConfusionMatrixLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs ) {
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

    matrix_ = new long double[classes_ * classes_];
    per_class_ = new long double[classes_];

    Reset();
  }

  return valid;
}

void ConfusionMatrixLayer::FeedForward() {
  if ( disabled_ )
    return;

  for ( unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    for ( unsigned int y = 0; y < first_->data.height(); y++ ) {
      for ( unsigned int x = 0; x < first_->data.width(); x++ ) {
        unsigned int first_class = first_->data.PixelMaximum ( x,y,sample );
        unsigned int second_class = second_->data.PixelMaximum ( x,y,sample );
        const long double weight = *third_->data.data_ptr_const ( x,y,0,sample );
        matrix_[ ( first_class * classes_ ) + second_class] += weight;
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
  for ( unsigned int c = 0; c < ( classes_ * classes_ ); c++ ) {
    matrix_[c] = 0;
  }

  for ( unsigned int c = 0; c < classes_; c++ ) {
    per_class_[c] = 0;
  }

  total_ = 0;
  right_ = 0;
}

void ConfusionMatrixLayer::Print ( std::string prefix, bool training ) {
  std::stringstream caption;
  caption << std::setw ( 12 ) << "vCLS  ACT>";

  for ( unsigned int c = 0; c < classes_; c++ ) {
    caption << std::setw ( 12 ) << names_[c];
  }

  (training?LOGTRESULT:LOGRESULT) << caption.str() << LOGRESULTEND;
  caption.str ( "" );


  for ( unsigned int t = 0; t < classes_; t++ ) {
    caption << std::setw ( 12 ) << names_[t];

    for ( unsigned int c = 0; c < classes_; c++ ) {
      long double result = matrix_[ ( t * classes_ ) + c];
      caption << std::setw ( 12 ) << static_cast<long> ( result );
    }

    (training?LOGTRESULT:LOGRESULT) << caption.str() << LOGRESULTEND;
    caption.str ( "" );
  }


  (training?LOGTRESULT:LOGRESULT) << prefix << " Overall recognition rate (not normalized): "
            << 100.0L * right_ / total_ << "%";

  long double ccount = 0;
  long double sum = 0;

  for ( unsigned int c = 0; c < classes_; c++ ) {
    if ( per_class_[c] > 0 ) {
      sum += matrix_[ ( c * classes_ ) + c] / per_class_[c];
      ccount += 1.0L;
    }
  }

  (training?LOGTRESULT:LOGRESULT) << prefix << " Average recognition rate (normalized)    : "
            << 100.0 * sum / ccount << "%" << LOGRESULTEND;

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
    caption << std::setw(12) << names_[t];
    caption << " IU: ";
    caption << std::setw(12) << IU * 100.0;
    caption << "%";
    (training?LOGTRESULT:LOGRESULT) << prefix << caption.str() << LOGRESULTEND;
    caption.str ( "" );
  }

  (training?LOGTRESULT:LOGRESULT) << prefix << " Average intersection over union          : "
            << 100.0 * IU_sum / (long double)classes_ << "%" << LOGRESULTEND;
}

ConfusionMatrixLayer::~ConfusionMatrixLayer() {
  if ( matrix_ != nullptr )
    delete[] matrix_;
}




}
