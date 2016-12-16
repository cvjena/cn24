/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_YOLODYNAMICOUTPUTLAYER_H
#define CONV_YOLODYNAMICOUTPUTLAYER_H

#include <random>
#include <fstream>

#include "SimpleLayer.h"
#include "../util/ClassManager.h"

namespace Conv {
class YOLODynamicOutputLayer : public SimpleLayer, public ClassManager::ClassUpdateHandler {
public:
  YOLODynamicOutputLayer(JSON configuration, ClassManager* class_manager);
  void UpdateTensorSizes(bool no_init = false);
  inline void OnClassUpdate() { UpdateTensorSizes(); }

  // Implementations for SimpleLayer
  bool Connect(const CombinedTensor *input, CombinedTensor *output);

  // Implementations for Layer
  bool CreateOutputs(const std::vector<CombinedTensor *> &inputs, std::vector<CombinedTensor *> &outputs);
  void FeedForward();
  void BackPropagate();
  std::string GetLayerDescription() { return "YOLO Dynamic Output Layer"; }
  void OnLayerConnect (const std::vector<Layer*> next_layer, bool no_init);

  bool Serialize(std::ostream& output_stream);
  bool Deserialize(unsigned int metadata_length, const char* metadata, unsigned int parameter_set_size, std::istream& input_stream);

  bool IsSerializationAware() { return true;  }

  inline unsigned int Gain() {
    return input_->data.maps();
  }
private:
  ClassManager* class_manager_;
  unsigned int horizontal_cells_ = 0;
  unsigned int vertical_cells_ = 0;
  unsigned int boxes_per_cell_ = 0;

  CombinedTensor* box_weights_;
  CombinedTensor* box_biases_;
  CombinedTensor* class_weights_;
  CombinedTensor* class_biases_;

  std::mt19937 rand_;

  unsigned int next_layer_gain_ = 0;
};
}

#endif
