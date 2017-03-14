/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#ifndef CONV_PREPROCESSING_LAYER_H
#define CONV_PREPROCESSING_LAYER_H

#include "SimpleLayer.h"

namespace Conv {
  
class PreprocessingLayer : public SimpleLayer {
public:
  explicit PreprocessingLayer(JSON configuration);
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  virtual void FeedForward();
  virtual void BackPropagate();

  virtual std::string GetLayerDescription() { return "Preprocessing Layer"; }
private:
  datum multiply = 1;
  datum subtract = 0;
  bool do_mean_subtraction = false;
  bool opencv_channel_swap = false;
  bool do_mean_image = false;
  int crop_x = 0;
  int crop_y = 0;
  
  CombinedTensor* mean_image_ = nullptr;
};

}

#endif