/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ShellState.h"

#include <fstream>

namespace Conv {
  
CN24_SHELL_FUNC_IMPL(PredictImage) {
  CN24_SHELL_FUNC_DESCRIPTION("Run prediction on a single image");
  
  char* file = nullptr;
  cargo_add_option(cargo, (cargo_option_flags_t)0, "file", "Image to predict",
    "s", &file);
  
  CN24_SHELL_PARSE_ARGS;
  
  if(state_ != NET_AND_TRAINER_LOADED && state_ != NET_LOADED) {
    LOGERROR << "Cannot predict image, no network is loaded.";
    return FAILURE;
  }
  
  std::string file_str = std::string(file);
  std::string path = PathFinder::FindPath(file_str,{});
  
  if(path.length() == 0) {
    LOGERROR << "Cannot find file: " << file;
    return FAILURE;
  }
  
  // Prepare sample
  JSON sample_json;
  sample_json["image_filename"] = file_str;
  sample_json["image_rpath"] = path;
  
  // Switch to testing mode, just in case
  graph_->SetIsTesting(true);
  
  // Load sample into net
  switch(input_layer_->GetTask()) {
    case DETECTION:
      sample_json["boxes"] = JSON::array();
      input_layer_->ForceLoadDetection(sample_json, 0);
      break;
    case CLASSIFICATION:
      input_layer_->ForceLoadClassification(sample_json, 0);
      break;
    default:
      LOGERROR << "Not implemented yet!";
      return FAILURE;
  }
  
  // Predict!
  graph_->FeedForward();
  
  // Get output
  Tensor& net_output = graph_->GetDefaultOutputNode()->output_buffers[0].combined_tensor->data;
#ifdef BUILD_OPENCL
  net_output.MoveToCPU();
#endif
  
  switch(input_layer_->GetTask()) {
    case DETECTION:
      {
        DetectionMetadataPointer net_metadata = (DetectionMetadataPointer)(graph_->GetDefaultOutputNode()->output_buffers[0].combined_tensor->metadata[0]);
        LOGINFO << "Detected " << net_metadata->size() << " objects:";
        for(unsigned int i = 0; i < net_metadata->size(); i++) {
          BoundingBox box = net_metadata->at(i);
          std::string class_name = class_manager_->GetClassInfoById(box.c).first;
          LOGINFO << "  Box " << i << ": " << class_name << " (" << box.score << ")";
        }
      }
      break;
    case CLASSIFICATION:
      {
        int highest_class = net_output.PixelMaximum(0,0,0);
        LOGINFO << "Highest class: " << highest_class << " " << class_manager_->GetClassInfoById(highest_class).first << " (" << net_output(highest_class) << ")";
      }
      break;
    default:
      LOGINFO << "Not implemented yet, but the prediction worked.";
      break;
  }
  
  return SUCCESS;
}
}