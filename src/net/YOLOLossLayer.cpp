/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cmath>
#include <cn24/util/BoundingBox.h>

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
  if ( inputs.size() != 3 ) {
    LOGERROR << "Need exactly 3 inputs to calculate loss function!";
    return false;
  }

  // Also, the two inputs have to have the same number of samples and elements!
  // We ignore the shape for now...
  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];
  CombinedTensor* third = inputs[2];
  bool valid = first != nullptr && second != nullptr &&
               outputs.size() == 0 && second->metadata != nullptr;
  if(!valid) {
    LOGERROR << "Failed null pointer and size check!";
  }

  unsigned int total_maps = first->data.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  classes_ = maps_per_cell - (5 * boxes_per_cell_);

  unsigned int should_be_maps = ((5 * boxes_per_cell_) + classes_) * horizontal_cells_ * vertical_cells_;

  if(should_be_maps != total_maps) {
    LOGERROR << "Wrong number of output maps detected! Should be " << total_maps << " (" << horizontal_cells_ << "x" << vertical_cells_ << "x(" << classes_ << "+5).";
  }

  valid &= should_be_maps == total_maps;

  if ( valid ) {
    first_ = first;
    second_ = second;
    third_ = third;
  }

  return valid;
}

void YOLOLossLayer::FeedForward() {
  unsigned int total_maps = (unsigned int)first_->data.maps();
  unsigned int maps_per_cell = total_maps / (horizontal_cells_ * vertical_cells_);
  classes_ = maps_per_cell - (5 * boxes_per_cell_);

  // We write the deltas at this point, because
  // CalculateLossFunction() is called before BackPropagate().
  //pragma omp parallel for default(shared)
  first_->delta.Clear((datum)0);
  current_loss_ = 0;

  for (unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    datum sample_weight = *(third_->data.data_ptr(0, 0, 0, sample));
    if(sample_weight == 0) {
      continue;
    } else if(sample_weight != 1) {
      FATAL("Unsupported sample weight: " << sample_weight);
    }
    std::vector<BoundingBox>* truth_boxes = (std::vector<BoundingBox>*)second_->metadata[sample];

    // Prepare indices into the prediction array
    unsigned int sample_index = (unsigned int)(sample * first_->data.maps());
    unsigned int class_index = sample_index + (vertical_cells_ * horizontal_cells_ * boxes_per_cell_ * 5);
    unsigned int confidence_index = sample_index;
    unsigned int coords_index = sample_index;

    // Loop over all cells
    for (unsigned int vcell = 0; vcell < vertical_cells_; vcell++) {
      for (unsigned int hcell = 0; hcell < horizontal_cells_; hcell++) {
        unsigned int cell_id = vcell * horizontal_cells_ + hcell;
        const datum box_xmin = ((datum)hcell) / (datum)horizontal_cells_;
        const datum box_ymin = ((datum)vcell) / (datum)vertical_cells_;
        const datum box_xmax = ((datum)(hcell + 1)) / (datum)horizontal_cells_;
        const datum box_ymax = ((datum)(vcell+ 1)) / (datum)vertical_cells_;

        // Find matching ground truth box
        BoundingBox* truth_box = nullptr;
        bool found_box = false;
        for(unsigned int t = 0; t < truth_boxes->size(); t++) {
          BoundingBox* tbox = &((*truth_boxes)[t]);
          if(tbox->x >= box_xmin && tbox->x < box_xmax
          && tbox->y >= box_ymin && tbox->y < box_ymax) {
            truth_box = tbox;
            found_box = true;
            break;
          }
        };

        datum best_iou = 0;
        int responsible_box = -1; // So that no possible index is equal to responsible_box when it is not overwritten

        // Loop over all possible boxes to find "responsible" box
        if(found_box) {
          for (unsigned int b = 0; b < boxes_per_cell_; b++) {
            unsigned int box_coords_index = coords_index + 5 * (boxes_per_cell_ * cell_id + b);

            // Calculate in-image coordinates
            const datum x = box_xmin + (first_->data.data_ptr_const()[box_coords_index] / (datum) horizontal_cells_);
            const datum y = box_ymin + (first_->data.data_ptr_const()[box_coords_index + 1] / (datum) vertical_cells_);
            const datum w = first_->data.data_ptr_const()[box_coords_index + 2] *
                            first_->data.data_ptr_const()[box_coords_index + 2];
            const datum h = first_->data.data_ptr_const()[box_coords_index + 3] *
                            first_->data.data_ptr_const()[box_coords_index + 3];

            // Calculate actual IOU
            BoundingBox box(x, y, w, h);
            datum actual_iou = box.IntersectionOverUnion(truth_box);

            if (actual_iou > best_iou) {
              responsible_box = b;
              best_iou = actual_iou;
            }
          }

          // Loop over all classes to assign loss for each
          for (unsigned int c = 0; c < classes_; c++) {
            unsigned int cell_class_index = class_index + (horizontal_cells_ * vertical_cells_ * c) + cell_id;
            const datum predicted_class_prob = first_->data.data_ptr_const()[cell_class_index];

            const datum class_delta = predicted_class_prob - (truth_box->c == c ? (datum)1.0 : (datum)0.0);
            first_->delta[cell_class_index] = (datum)2.0 * class_delta;
            current_loss_ += (class_delta * class_delta);
          }
        }

        // Loop over all boxes to calculate loss
        for (unsigned int b = 0; b < boxes_per_cell_; b++) {
          // Box predicted iou is needed for both cases
          unsigned int box_confidence_index = confidence_index + (cell_id * boxes_per_cell_ + b) * 5 + 4;
          const datum box_confidence = first_->data.data_ptr_const()[box_confidence_index];

          if(((int)b) == responsible_box) {
            // Box b is "responsible" for the detection
            unsigned int box_coords_index = coords_index + 5 * (boxes_per_cell_ * cell_id + b);

            // Calculate in-image coordinates
            const datum x = box_xmin + (first_->data.data_ptr_const()[box_coords_index] / (datum) horizontal_cells_);
            const datum y = box_ymin + (first_->data.data_ptr_const()[box_coords_index + 1] / (datum) vertical_cells_);
            const datum w = first_->data.data_ptr_const()[box_coords_index + 2];
            const datum h = first_->data.data_ptr_const()[box_coords_index + 3];

            // Calculate actual IOU
            BoundingBox box(x, y, w * w, h * h);
            datum actual_iou = box.IntersectionOverUnion(truth_box);

            // Loss: Box coordinates
            const datum xcoord_delta = (x - truth_box->x) * (datum)horizontal_cells_;
            const datum ycoord_delta = (y - truth_box->y) * (datum)vertical_cells_;
            first_->delta.data_ptr()[box_coords_index] = scale_coord_ * (datum)2.0 * xcoord_delta;
            first_->delta.data_ptr()[box_coords_index + 1] = scale_coord_ * (datum)2.0 * ycoord_delta;
            current_loss_ += (datum)(scale_coord_ * xcoord_delta * xcoord_delta) + (datum)(scale_coord_ * ycoord_delta * ycoord_delta);

            // Loss: Box size
            const datum w_delta = w - std::sqrt(truth_box->w);
            const datum h_delta = h - std::sqrt(truth_box->h);
            first_->delta.data_ptr()[box_coords_index + 2] = scale_coord_ * (datum)2.0 * w_delta;
            first_->delta.data_ptr()[box_coords_index + 3] = scale_coord_ * (datum)2.0 * h_delta;
            current_loss_ += (datum)(scale_coord_ * w_delta * w_delta) + (datum)(scale_coord_ * h_delta * h_delta);

            // Loss: Predicted confidence
            const datum conf_delta = box_confidence - actual_iou;
            current_loss_ += (datum)(conf_delta * conf_delta);
            first_->delta.data_ptr()[box_confidence_index] = (datum)2.0 * conf_delta;
          } else {
            // Box b is not "responsible" for the ground truth

            // Loss: Box confidence
            first_->delta.data_ptr()[box_confidence_index] = scale_noobj_* ((datum)2.0 * box_confidence);
            current_loss_ += (datum)(scale_noobj_ * box_confidence * box_confidence);
          }
        }

      }
    }
  }
}

void YOLOLossLayer::BackPropagate() {
  // The deltas are already written in to the input CombinedTensors, so
  // there is nothing to do now.
}

datum YOLOLossLayer::CalculateLossFunction() {
  return (datum)current_loss_;
}


}
