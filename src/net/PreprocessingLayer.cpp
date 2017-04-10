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
  JSON_TRY_BOOL(opencv_channel_swap, configuration, "opencv_channel_swap", false);
  JSON_TRY_BOOL(do_mean_image, configuration, "mean_image", false);
  
  if(configuration.count("crop") == 1 && configuration["crop"].is_array() && configuration["crop"].size() == 2) {
    LOGDEBUG << "Cropping!";
    crop_x = configuration["crop"][0];
    crop_y = configuration["crop"][1];
  }
  
  LOGDEBUG << "Initialized! M: " << multiply << ", S:" << subtract << ", SM: "
  << do_mean_subtraction << ", SW: " << opencv_channel_swap;
  LOGDEBUG << "Crop: " << crop_x << "," << crop_y;
  if(do_mean_image) {
    LOGDEBUG << "Subtracting mean image";
  }
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

  if(crop_x == 0 && crop_y == 0) {
    crop_x = input->data.width();
    crop_y = input->data.height();
  }
  
  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      crop_x,
      crop_y,
      input->data.maps());

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool PreprocessingLayer::Connect(const CombinedTensor* input, CombinedTensor* output)
{
  // Check dimensions for equality
  bool valid = input->data.samples() == output->data.samples() &&
               input->data.maps() == output->data.maps();

  if(do_mean_image) {
    mean_image_ = new CombinedTensor(input->data.samples(), input->data.width(),
                                    input->data.height(), input->data.maps(),
                                    nullptr, false, "PreprocessingLayer");
    
    parameters_.push_back(mean_image_);
  }

  return valid;
}

void PreprocessingLayer::FeedForward()
{
  const unsigned int elements_per_map = input_->data.elements() / (input_->data.maps() * input_->data.samples());
  unsigned int crop_offset_x = floor((input_->data.width() - crop_x) / 2) + 1;
  unsigned int crop_offset_y = floor((input_->data.height() - crop_y) / 2) + 1;
  if(crop_x == input_->data.width())
    crop_offset_x = 0;
  if(crop_y == input_->data.height())
    crop_offset_y = 0;

  // 1. Multiplication and subtraction (with channel swap if needed)
  for(unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    for(unsigned int map = 0; map < input_->data.maps(); map++) {
      unsigned int source_map = opencv_channel_swap ? 2 - map : map;
      for(unsigned int y = 0; y < crop_y; y++) {
        for(unsigned int x = 0; x < crop_x; x++) {
          *(output_->data.data_ptr(x,y,map,sample)) =
            *(input_->data.data_ptr_const(x + crop_offset_x, y + crop_offset_y,
              source_map, sample)) * multiply - subtract;
              
          if(do_mean_image) {
            const datum mean_sub = 
              *(mean_image_->data.data_ptr_const(x + crop_offset_x, y + crop_offset_y,
                map, 0));
            *(output_->data.data_ptr(x,y,map,sample)) -= mean_sub;
          }
        }
      }
    }
  }
 
  // 2. Mean subtraction
  if(do_mean_subtraction) {
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