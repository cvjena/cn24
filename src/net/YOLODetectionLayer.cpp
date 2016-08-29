/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <vector>

#include "CombinedTensor.h"
#include "Log.h"
#include "BoundingBox.h"

#include "YOLODetectionLayer.h"

namespace Conv {

YOLODetectionLayer::YOLODetectionLayer(JSON configuration): SimpleLayer(configuration) {
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
  
bool YOLODetectionLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs) {
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

  output->data.Shadow(input->data);
  output->delta.Shadow(input->delta);

  DatasetMetadataPointer* metadata_buffer = new DatasetMetadataPointer[input->data.samples()];
  output->metadata = metadata_buffer;

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool YOLODetectionLayer::Connect (const CombinedTensor* input,
                                 CombinedTensor* output) {
  // Check dimensions for equality
  bool valid = input->data.samples() == output->data.samples() &&
               input->data.width() == output->data.width() &&
               input->data.height() == output->data.height() &&
               input->data.maps() == output->data.maps();

  valid &= output->metadata != nullptr;

  unsigned int total_maps = input->data.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  classes_ = maps_per_cell - (5 * boxes_per_cell_);

  unsigned int should_be_maps = ((5 * boxes_per_cell_) + classes_) * horizontal_cells_ * vertical_cells_;

  if(should_be_maps != total_maps) {
    LOGERROR << "Wrong number of output maps detected! Should be " << total_maps << " (" << horizontal_cells_ << "x" << vertical_cells_ << "x(" << classes_ << "+5).";
  }

  valid &= should_be_maps == total_maps;

  // TODO Test correct shadowing

  if(valid) {
    metadata_buffer_ = output->metadata;
    for(unsigned int s = 0; s < input->data.samples(); s++) {
      std::vector<BoundingBox>* v = new std::vector<BoundingBox>;
      metadata_buffer_[s] = (DatasetMetadataPointer)v;
    }
  }

  return valid;
}

void YOLODetectionLayer::FeedForward() {
  unsigned int maps_per_cell = input_->data.maps() / (horizontal_cells_ * vertical_cells_);
  for (unsigned int sample = 0; sample < input_->data.samples(); sample++ ) {
    unsigned int sample_index = sample * input_->data.maps();
    LOGDEBUG << "Processing sample " << sample;
    unsigned int class_index = sample_index;
    unsigned int iou_index = sample_index + (classes_ * vertical_cells_ * horizontal_cells_);
    unsigned int coords_index = iou_index + vertical_cells_ * horizontal_cells_ * boxes_per_cell_;
    for (unsigned int vcell = 0; vcell < vertical_cells_; vcell++) {
      for (unsigned int hcell = 0; hcell < horizontal_cells_; hcell++) {
        unsigned int cell_id = vcell * horizontal_cells_ + hcell;
        for (unsigned int b = 0; b < boxes_per_cell_; b++) {
          const datum iou = input_->data.data_ptr_const()[iou_index + cell_id * boxes_per_cell_ + b];
          unsigned int box_coords_index = coords_index + 4 * (boxes_per_cell_ * cell_id + b);
          if(iou > 0.2) {
            LOGDEBUG << "Cell (" << hcell << "," << vcell << "), box " << b << " iou: " << iou;
            // Show coordinates
            const datum x = input_->data.data_ptr_const()[box_coords_index];
            const datum y = input_->data.data_ptr_const()[box_coords_index + 1];
            const datum w = input_->data.data_ptr_const()[box_coords_index + 2];
            const datum h = input_->data.data_ptr_const()[box_coords_index + 3];

            LOGDEBUG << "Cell (" << hcell << "," << vcell << "), box " << b << " at : (" << x << "," << y << "), size (" << w * w << "," << h * h << ")";

            // Show classes
            for (unsigned int i = 0; i < classes_; i++) {
              LOGDEBUG << "Cell (" << hcell << "," << vcell << "), class " << i << ": " << input_->data.data_ptr_const()[class_index + cell_id * classes_ + i];
            }
          }
        }
      }

    }

    std::vector<BoundingBox>* sample_boxes = (std::vector<BoundingBox>*)output_->metadata[sample];
    LOGDEBUG << "  " << sample_boxes->size() << " bounding boxes in output";
  }

}

void YOLODetectionLayer::BackPropagate() {

}


}
