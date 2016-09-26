/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "YOLODynamicOutputLayer.h"

namespace Conv {

YOLODynamicOutputLayer::YOLODynamicOutputLayer(JSON configuration, ClassManager *class_manager) :
  SimpleLayer(configuration), class_manager_(class_manager) {
  if(configuration.count("yolo_configuration") != 1) {
    FATAL("YOLO configuration missing!");
  }

  JSON yolo_configuration = configuration["yolo_configuration"];
  if(yolo_configuration.count("horizontal_cells") != 1 || !yolo_configuration["horizontal_cells"].is_number()) {
    FATAL("YOLO yolo_configuration property horizontal_cells missing!");
  }
  horizontal_cells_ = yolo_configuration["horizontal_cells"];

  if(yolo_configuration.count("vertical_cells") != 1 || !yolo_configuration["vertical_cells"].is_number()) {
    FATAL("YOLO yolo_configuration property vertical_cells missing!");
  }
  vertical_cells_ = yolo_configuration["vertical_cells"];

  if(yolo_configuration.count("boxes_per_cell") != 1 || !yolo_configuration["boxes_per_cell"].is_number()) {
    FATAL("YOLO yolo_configuration property boxes_per_cell missing!");
  }
  boxes_per_cell_ = yolo_configuration["boxes_per_cell"];

}

void YOLODynamicOutputLayer::UpdateTensorSizes() {
  unsigned int class_maps = horizontal_cells_ * vertical_cells_ * (class_manager_->GetMaxClassId() + 1);
  unsigned int output_maps = horizontal_cells_ * vertical_cells_ * (boxes_per_cell_ * 5 + (class_manager_->GetMaxClassId() + 1));

  if(output_->data.maps() != output_maps) {
    output_->data.Resize(input_->data.samples(), 1, 1, output_maps);
    output_->delta.Resize(input_->data.samples(), 1, 1, output_maps);
  }

  if(class_weights_->data.samples() != class_maps) {
    class_weights_->data.Extend(class_maps);
    class_weights_->delta.Extend(class_maps);
    class_biases_->data.Extend(class_maps);
    class_biases_->delta.Extend(class_maps);
  }
}

bool YOLODynamicOutputLayer::Connect(const CombinedTensor *input, CombinedTensor *output) {
  if(output == nullptr || input == nullptr) {
    LOGERROR << "Null-pointer node supplied!";
    return false;
  }
  unsigned int output_maps = horizontal_cells_ * vertical_cells_ * (boxes_per_cell_ * 5 + (class_manager_->GetMaxClassId() + 1));

  box_weights_ = new CombinedTensor(horizontal_cells_ * vertical_cells_ * boxes_per_cell_ * 5, 1, 1, input->data.maps());
  box_weights_->is_dynamic = true;
  box_biases_ = new CombinedTensor(horizontal_cells_ * vertical_cells_ * boxes_per_cell_ * 5, 1, 1, 1);
  box_biases_->is_dynamic = true;

  class_weights_ = new CombinedTensor(horizontal_cells_ * vertical_cells_ * (class_manager_->GetMaxClassId() + 1), 1, 1, input->data.maps());
  class_weights_->is_dynamic = true;
  class_biases_ = new CombinedTensor(horizontal_cells_ * vertical_cells_ * (class_manager_->GetMaxClassId() + 1), 1, 1, 1);
  class_biases_->is_dynamic = true;

  parameters_.push_back(box_weights_);
  parameters_.push_back(box_biases_);
  parameters_.push_back(class_weights_);
  parameters_.push_back(class_biases_);


  class_manager_->RegisterClassUpdateHandler(this);
  return output->data.width() == 1 && output->data.height() == 1 && output->data.samples() == input->data.samples() && output->data.maps() == output_maps;
}

bool YOLODynamicOutputLayer::CreateOutputs(const std::vector<CombinedTensor *> &inputs,
                                           std::vector<CombinedTensor *> &outputs) {
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
  unsigned int output_maps = horizontal_cells_ * vertical_cells_ * (boxes_per_cell_ * 5 + (class_manager_->GetMaxClassId() + 1));
  CombinedTensor* output = new CombinedTensor (input->data.samples(), 1, 1, output_maps);
  output->is_dynamic = true;

  outputs.push_back(output);
  return true;
}

void YOLODynamicOutputLayer::FeedForward() {
  UpdateTensorSizes();
}

void YOLODynamicOutputLayer::BackPropagate() {

}
}