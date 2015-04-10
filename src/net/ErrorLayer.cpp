/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cmath>

#include "Log.h"
#include "CombinedTensor.h"

#include "ErrorLayer.h"

namespace Conv {

ErrorLayer::ErrorLayer() {
  LOGDEBUG << "Instance created.";
#ifdef ERROR_LAYER_IGNORE_WEIGHTS
  LOGINFO << "Weights are being ignored!";
#endif
}

bool ErrorLayer::CreateOutputs ( const std::vector< CombinedTensor* >& inputs,
                                 std::vector< CombinedTensor* >& outputs ) {
  // Validate input node count
  if ( inputs.size() != 3 ) {
    LOGERROR << "Need exactly 3 inputs to calculate loss function!";
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

bool ErrorLayer::Connect ( const std::vector< CombinedTensor* >& inputs,
                           const std::vector< CombinedTensor* >& outputs,
                           const NetStatus* net ) {
  // Needs exactly three inputs to calculate the difference
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

void ErrorLayer::FeedForward() {
  // We write the deltas at this point, because
  // CalculateLossFunction() is called before BackPropagate().
  // We don't precalculate the loss because it is not calculated for every
  // batch.
  //pragma omp parallel for default(shared)
  for ( unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    for ( unsigned int map = 0; map < first_->data.maps(); map++ ) {
      for ( unsigned int y = 0; y < first_->data.height(); y++ ) {
        for ( unsigned int x = 0; x < first_->data.width(); x++ ) {
          const datum first =
            *first_->data.data_ptr_const ( x, y, map, sample );
          const datum second =
            *second_->data.data_ptr_const ( x, y, map, sample );

          const datum diff = first - second;

#ifdef ERROR_LAYER_IGNORE_WEIGHTS
          const datum weight = 1.0;
#else
		  const datum weight =
            *third_->data.data_ptr_const ( x,y,0,sample );
#endif
	  *first_->delta.data_ptr(x,y,map,sample) = diff * weight;
        }
      }
    }
  }


/*  for ( std::size_t i = 0; i < first_->data.elements(); i++ ) {
    const datum first = first_->data.data_ptr_const() [i];
    const datum second = second_->data.data_ptr_const() [i];
    const datum diff = first_->data.data_ptr_const() [i] -
                       second_->data.data_ptr_const() [i];
#ifdef ERROR_LAYER_IGNORE_WEIGHTS
    const datum weight = 1.0;
#else
    const datum weight = third_->data.data_ptr_const() [i];
#endif
    first_->delta.data_ptr() [i] = diff * weight;
  }
*/
}

void ErrorLayer::BackPropagate() {
  // The deltas are already written in to the input CombinedTensors, so
  // there is nothing to do now.
}

datum ErrorLayer::CalculateLossFunction() {
  long double error = 0;

  // Add up the squared error
  for (unsigned int sample = 0; sample < first_->data.samples(); sample++) {
	  for (unsigned int map = 0; map < first_->data.maps(); map++) {
		  for (unsigned int y = 0; y < first_->data.height(); y++) {
			  for (unsigned int x = 0; x < first_->data.width(); x++) {
				  const datum first =
					  *first_->data.data_ptr_const(x, y, map, sample);
				  const datum second =
					  *second_->data.data_ptr_const(x, y, map, sample);

				  const datum diff = first - second;

#ifdef ERROR_LAYER_IGNORE_WEIGHTS
				  const datum weight = 1.0;
#else
				  const datum weight = 
					*third_->data.data_ptr_const ( x,y,0,sample );
#endif
				  error += ((long double)diff) * ((long double)diff) * ((long double)weight);
			  }
		  }
	  }
  }

  return error / 2.0;
}


}
