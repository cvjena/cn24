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
  char* out_file = nullptr;
  cargo_add_option(cargo, (cargo_option_flags_t)0, "file", "Image to predict",
    "s", &file);
  cargo_add_option(cargo, (cargo_option_flags_t)CARGO_OPT_NOT_REQUIRED, "outfile",
    "Output image (for binary segmentation, detection)", "s", &out_file);
  
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

  bool load_result = true;
  // Load sample into net
  switch(input_layer_->GetTask()) {
    case DETECTION:
      sample_json["boxes"] = JSON::array();
      load_result = input_layer_->ForceLoadDetection(sample_json, 0);
      break;
    case BINARY_SEGMENTATION:
      load_result = input_layer_->ForceLoadBinarySegmentation(sample_json, 0);
      break;
    case CLASSIFICATION:
      load_result = input_layer_->ForceLoadClassification(sample_json, 0);
      break;
    default:
      LOGERROR << "Not implemented yet!";
      return FAILURE;
  }

  if(!load_result) {
    return FAILURE;
  }

  // Predict!
  input_layer_->ForceWeightsZero();
  graph_->FeedForward();
  
  // Get output
  Tensor& net_output = graph_->GetDefaultOutputNode()->output_buffers[0].combined_tensor->data;
#ifdef BUILD_OPENCL
  net_output.MoveToCPU();
#endif

  // Console output
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

  // Graphical (file) output
  if(out_file != nullptr) {
    std::string out_file_str = out_file;

    // Get net input
    Tensor out_file_tensor = Tensor(net_input()->data, true, "Shell.Predict");

    switch(input_layer_->GetTask()) {
      case DETECTION: {
        for (unsigned int m = 0; m < out_file_tensor.maps(); m++) {
          DetectionMetadataPointer net_metadata = (DetectionMetadataPointer) (graph_->GetDefaultOutputNode()->output_buffers[0].combined_tensor->metadata[0]);
          for (unsigned int i = 0; i < net_metadata->size(); i++) {
            BoundingBox box = net_metadata->at(i);

            // Calculate bounding coordinates and denormalize
            int left = (int) ((box.x - box.w / 2.0f) * out_file_tensor.width());
            int right = (int) ((box.x + box.w / 2.0f) * out_file_tensor.width());
            int top = (int) ((box.y - box.h / 2.0f) * out_file_tensor.height());
            int bottom = (int) ((box.y + box.h / 2.0f) * out_file_tensor.height());

            // Clamp values
            if (left < 0) left = 0; if (left >= out_file_tensor.width()) left = (int) out_file_tensor.width() - 1;
            if (right < 0) right = 0; if (right >= out_file_tensor.width()) right = (int) out_file_tensor.width() - 1;
            if (top < 0) top = 0; if (top >= out_file_tensor.height()) top = (int) out_file_tensor.height() - 1;
            if (bottom < 0) bottom = 0; if (bottom >= out_file_tensor.height()) bottom = (int) out_file_tensor.height() - 1;

            // Draw lines
            for (unsigned int x = (unsigned) left; x <= right; x++) {
              out_file_tensor.data_ptr(x, top, m, 0)[0] = 1;
              out_file_tensor.data_ptr(x, bottom, m, 0)[0] = 1;
            }
            for (unsigned int y = (unsigned) top; y <= bottom; y++) {
              out_file_tensor.data_ptr(left, y, m, 0)[0] = 1;
              out_file_tensor.data_ptr(right, y, m, 0)[0] = 1;
            }
          }
        }
      }
        break;
      case BINARY_SEGMENTATION: {
        // Try to use a channel that is actually in the image
        int channel = 1;
        if (out_file_tensor.maps() < 3)
          channel = out_file_tensor.maps() - 1;

        for (unsigned int y = 0; y < net_output.height(); y++) {
          for (unsigned int x = 0; x < net_output.width(); x++) {
            datum value = net_output.data_ptr_const(x, y, 0, 0)[0];

            // ASSUME sigmoid output
            datum original_value = out_file_tensor.data_ptr_const(x, y, channel, 0)[0];
            datum new_value = (1.0f - value) * original_value + (value) * 1.0f;

            out_file_tensor.data_ptr(x, y, channel, 0)[0] = new_value;
          }
        }
      }
        break;
      default:
        LOGINFO << "Not implemented yet, but the prediction worked.";
        break;
    }

    out_file_tensor.WriteToFile(out_file_str);
  }

  return SUCCESS;
}
}
