/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <algorithm>

#include "YOLODynamicOutputLayer.h"

namespace Conv {

YOLODynamicOutputLayer::YOLODynamicOutputLayer(JSON configuration, ClassManager *class_manager) :
  SimpleLayer(configuration), class_manager_(class_manager) {
  unsigned int seed = 0;

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

  if(configuration.count("seed") == 1 && configuration["seed"].is_number()) {
    seed = configuration["seed"];
  }

  rand_.seed(seed);
}

void YOLODynamicOutputLayer::UpdateTensorSizes() {
  unsigned int class_maps = horizontal_cells_ * vertical_cells_ * (class_manager_->GetMaxClassId() + 1);
  unsigned int output_maps = horizontal_cells_ * vertical_cells_ * (boxes_per_cell_ * 5 + (class_manager_->GetMaxClassId() + 1));

  if(output_->data.maps() != output_maps) {
    output_->data.Resize(input_->data.samples(), 1, 1, output_maps);
    output_->delta.Resize(input_->data.samples(), 1, 1, output_maps);
  }

  if(class_weights_->data.samples() != class_maps) {
    unsigned int old_class_maps = class_weights_->data.samples();
    if(class_maps < old_class_maps) {
      FATAL("This can never happen!");
    }

    class_weights_->data.Extend(class_maps);
    class_weights_->delta.Extend(class_maps);
    class_biases_->data.Extend(class_maps);
    class_biases_->delta.Extend(class_maps);

    // Randomly initialize new classes
    unsigned int this_layer_gain = Gain();

    const datum range = sqrt (6) / sqrt (next_layer_gain_ + this_layer_gain);
    std::uniform_real_distribution<datum> dist_weights (-range , range);

    for (std::size_t i = old_class_maps * input_->data.maps(); i < class_weights_->data.elements(); i++) {
      class_weights_->data[i] = dist_weights (rand_);
    }
    for (std::size_t i = old_class_maps; i < class_biases_->data.elements(); i++) {
      class_biases_->data[i] = 0; // dist_weights (rand_);
    }

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

  if(input->data.width() != 1 || input->data.height() != 1) {
    LOGERROR << "Input needs to be 1x1";
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

  const unsigned int class_offset = vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5;
  const unsigned int input_maps = input_->data.maps();
  const unsigned int box_output_maps = vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5;
  const unsigned int class_output_maps = vertical_cells_ * horizontal_cells_ * (class_manager_->GetMaxClassId() + 1);

  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    const datum* input_ptr = input_->data.data_ptr_const(0,0,0,sample);
    datum* output_ptr = output_->data.data_ptr(0,0,0,sample);

    // Box weights
    for (unsigned int m = 0; m < box_output_maps; m++) {
      datum sum = 0;
      for (unsigned int i = 0; i < input_maps; i++) {
        sum += input_ptr[i] * *(box_weights_->data.data_ptr_const(0, 0, i, m));
      }
      output_ptr[m] = sum;
    }

    // Class weights
    for (unsigned int m = 0; m < class_output_maps; m++) {
      datum sum = 0;
      for (unsigned int i = 0; i < input_maps; i++) {
        sum += input_ptr[i] * *(class_weights_->data.data_ptr_const(0, 0, i, m));
      }
      output_ptr[m + class_offset] = sum;
    }

    // Box biases
    for (unsigned int m = 0; m < box_biases_->data.samples(); m++) {
      output_ptr[m] += box_biases_->data.data_ptr_const()[m];
    }

    // Class biases
    for (unsigned int m = 0; m < class_biases_->data.samples(); m++) {
      output_ptr[m + class_offset] += class_biases_->data.data_ptr_const()[m];
    }
  }
}

void YOLODynamicOutputLayer::OnLayerConnect(const std::vector<Layer *> next_layers) {
  unsigned int next_layer_gain = 0;
  for (Layer* next_layer: next_layers)
    next_layer_gain += next_layer->Gain();

  unsigned int this_layer_gain = Gain();

  const datum range = sqrt (6) / sqrt (next_layer_gain + this_layer_gain);
  std::uniform_real_distribution<datum> dist_weights (-range , range);

  for (std::size_t i = 0; i < box_weights_->data.elements(); i++) {
    box_weights_->data[i] = dist_weights (rand_);
  }
  for (std::size_t i = 0; i < box_biases_->data.elements(); i++) {
    box_biases_->data[i] = 0; // dist_weights (rand_);
  }
  for (std::size_t i = 0; i < class_weights_->data.elements(); i++) {
    class_weights_->data[i] = dist_weights (rand_);
  }
  for (std::size_t i = 0; i < class_biases_->data.elements(); i++) {
    class_biases_->data[i] = 0; //dist_weights (rand_);
  }

  // Save gains of next layers for extension on class update
  next_layer_gain_ = next_layer_gain;
}

void YOLODynamicOutputLayer::BackPropagate() {
  const unsigned int class_offset = vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5;
  const unsigned int input_maps = input_->data.maps();
  const unsigned int box_output_maps = vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5;
  const unsigned int class_output_maps = vertical_cells_ * horizontal_cells_ * (class_manager_->GetMaxClassId() + 1);

  box_weights_->delta.Clear();
  box_biases_->delta.Clear();
  class_weights_->delta.Clear();
  class_biases_->delta.Clear();

  for (unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    const datum *input_data_ptr = input_->data.data_ptr_const(0, 0, 0, sample);
    datum *input_delta_ptr = input_->delta.data_ptr(0, 0, 0, sample);
    const datum *output_delta_ptr = output_->delta.data_ptr_const(0, 0, 0, sample);

    // Weight gradients
    // Box weights
    for (unsigned int m = 0; m < box_output_maps; m++) {
      for (unsigned int i = 0; i < input_maps; i++) {
        *box_weights_->delta.data_ptr(0, 0, i, m) += (output_delta_ptr[m] * input_data_ptr[i]);
      }
    }
    // Class weights
    for (unsigned int m = 0; m < class_output_maps; m++) {
      for (unsigned int i = 0; i < input_maps; i++) {
        *class_weights_->delta.data_ptr(0, 0, i, m) += (output_delta_ptr[m + class_offset] * input_data_ptr[i]);
      }
    }
    // Box biases
    for (unsigned int m = 0; m < box_output_maps; m++) {
      box_biases_->delta.data_ptr()[m] += output_delta_ptr[m];
    }
    // Class biases
    for (unsigned int m = 0; m < class_output_maps; m++) {
      class_biases_->delta.data_ptr()[m] += output_delta_ptr[m + class_offset];
    }

    // Input gradients
    for(unsigned int i = 0; i < input_maps; i++) {
      datum delta = 0;
      // Box weights
      for (unsigned int m = 0; m < box_output_maps; m++) {
        delta += output_delta_ptr[m] * *(box_weights_->data.data_ptr(0, 0, i, m));
      }
      // Class weights
      for (unsigned int m = 0; m < class_output_maps; m++) {
        delta += output_delta_ptr[m + class_offset] * *(class_weights_->data.data_ptr(0, 0, i, m));
      }
      input_delta_ptr[i] = delta;
    }

  }

}
}