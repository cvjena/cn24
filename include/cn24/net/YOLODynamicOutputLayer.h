/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_YOLODYNAMICOUTPUTLAYER_H
#define CONV_YOLODYNAMICOUTPUTLAYER_H

#include "SimpleLayer.h"
#include "ClassManager.h"

namespace Conv {
class YOLODynamicOutputLayer : public SimpleLayer {
public:
  YOLODynamicOutputLayer(JSON configuration, ClassManager* class_manager);
  void UpdateTensorSizes();

  // Implementations for SimpleLayer
  bool Connect(const CombinedTensor *input, CombinedTensor *output);

  // Implementations for Layer
  bool CreateOutputs(const std::vector<CombinedTensor *> &inputs, std::vector<CombinedTensor *> &outputs);
  void FeedForward();
  void BackPropagate();
  std::string GetLayerDescription() { return "YOLO Dynamic Output Layer"; }

private:
  ClassManager* class_manager_;
  unsigned int horizontal_cells_ = 0;
  unsigned int vertical_cells_ = 0;
  unsigned int boxes_per_cell_ = 0;

  CombinedTensor* box_weights_;
  CombinedTensor* box_biases_;
  CombinedTensor* class_weights_;
  CombinedTensor* class_biases_;
};
}

#endif
