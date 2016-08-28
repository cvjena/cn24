/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cmath>

#include "Log.h"
#include "CombinedTensor.h"

#include "YOLOLossLayer.h"

namespace Conv {

YOLOLossLayer::YOLOLossLayer(JSON configuration)
 : Layer(configuration) {
  LOGDEBUG << "Instance created.";
  if(configuration.count("horizontal_cells") != 1 || !configuration["horizontal_cells"].is_number()) {
    FATAL("YOLO configuration property horizontal_cells missing!");
  }
  horizontal_cells_ = configuration["horizontal_cells"];

  if(configuration.count("vertical_cells") != 1 || !configuration["vertical_cells"].is_number()) {
    FATAL("YOLO configuration property vertical_cells missing!");
  }
  vertical_cells_ = configuration["vertical_cells"];

  if(configuration.count("boxes_per_cell") != 1 || !configuration["boxes_per_cell"].is_number()) {
    FATAL("YOLO configuration property boxes_per_cell missing!");
  }
  boxes_per_cell_ = configuration["boxes_per_cell"];
}

bool YOLOLossLayer::CreateOutputs ( const std::vector< CombinedTensor* >& inputs,
                                 std::vector< CombinedTensor* >& outputs ) {
  UNREFERENCED_PARAMETER(outputs);
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

  // Needs no outputs
  return true;
}

bool YOLOLossLayer::Connect ( const std::vector< CombinedTensor* >& inputs,
                           const std::vector< CombinedTensor* >& outputs,
                           const NetStatus* net ) {
  UNREFERENCED_PARAMETER(net);
  // Needs exactly three inputs to calculate the difference
  if ( inputs.size() != 3 )
    return false;

  // Also, the two inputs have to have the same number of samples and elements!
  // We ignore the shape for now...
  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];
  CombinedTensor* third = inputs[2];
  bool valid = first != nullptr && second != nullptr &&
               outputs.size() == 0;

  if ( valid ) {
    first_ = first;
    second_ = second;
    third_ = third;
  }

  return valid;
}

void YOLOLossLayer::FeedForward() {
  // We write the deltas at this point, because
  // CalculateLossFunction() is called before BackPropagate().
  // We don't precalculate the loss because it is not calculated for every
  // batch.
  //pragma omp parallel for default(shared)
  for ( unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    LOGDEBUG << "Processing sample " << sample;
  }
  first_->delta.Clear();
}

void YOLOLossLayer::BackPropagate() {
  // The deltas are already written in to the input CombinedTensors, so
  // there is nothing to do now.
}

datum YOLOLossLayer::CalculateLossFunction() {
  long double error = 0;

  for (unsigned int sample = 0; sample < first_->data.samples(); sample++) {
				  error += 0;
  }
  return (datum)error;
}


}
