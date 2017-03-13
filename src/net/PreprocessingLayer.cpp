/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include "PreprocessingLayer.h"
#include "JSONParsing.h"
#include <TensorMath.h>

namespace Conv {
  
PreprocessingLayer::PreprocessingLayer(JSON configuration):
SimpleLayer(configuration)
{
  JSON_TRY_DATUM(multiply, configuration, "multiply_by", 1);
  JSON_TRY_DATUM(subtract, configuration, "subtract", 0);
  JSON_TRY_BOOL(do_mean_subtraction, configuration, "subtract_mean", false);
  
  LOGINFO << "Initialized! M: " << multiply << ", S:" << subtract << ", SM: "
  << do_mean_subtraction;
}

bool PreprocessingLayer::CreateOutputs(const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs)
{
  // This is a simple layer, only one input
  if (inputs.size() != 1) {
    LOGERROR << "Only one input supported!";
    return false;
  }

  // Save input node pointer
  CombinedTensor* input = inputs[0];

  // Check if input node pointer is null
  if (input == nullptr) {
    LOGERROR << "Null pointer input node!";
    return false;
  }

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      input->data.width(),
      input->data.height(),
      input->data.maps());

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool PreprocessingLayer::Connect(const CombinedTensor* input, CombinedTensor* output)
{
  // Check dimensions for equality
  bool valid = input->data.samples() == output->data.samples() &&
               input->data.width() == output->data.width() &&
               input->data.height() == output->data.height() &&
               input->data.maps() == output->data.maps();

  return valid;
}

void PreprocessingLayer::FeedForward()
{
  // 1. Multiplication and subtraction
#pragma omp parallel for default(shared)
  for(unsigned int e = 0; e < input_->data.elements(); e++) {
    output_->data[e] = input_->data[e] * multiply - subtract;
  }
  
  // 3. Mean subtraction
  if(do_mean_subtraction) {
    const unsigned int elements_per_map = input_->data.elements() / (input_->data.maps() * input_->data.samples());
#pragma omp parallel for default(shared)
    for(unsigned int sample = 0; sample < input_->data.samples(); sample++) {
      for(unsigned int map = 0; map < input_->data.maps(); map++) {
        datum map_mean = 0;
        datum* map_begin = output_->data.data_ptr(0, 0, map, sample);
        for(unsigned int e = 0; e < elements_per_map; e++) {
          map_mean += map_begin[e];
        }
        map_mean /= (datum)elements_per_map;
        for(unsigned int e = 0; e < elements_per_map; e++) {
          map_begin[e] -= map_mean;
        }
      }
    }
  }
}


void PreprocessingLayer::BackPropagate()
{

}


}