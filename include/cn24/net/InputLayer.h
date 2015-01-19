/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file InputLayer.h
 * \class InputLayer
 * \brief A simple layer that always outputs the same Tensor.
 * 
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_INPUTLAYER_H
#define CONV_INPUTLAYER_H

#include "Log.h"
#include "Tensor.h"
#include "CombinedTensor.h"
#include "Layer.h"

namespace Conv {
  
class InputLayer: public Layer {
public:
  explicit InputLayer(Tensor& data);
  InputLayer(Tensor& data, Tensor& helper); 
  InputLayer(Tensor& data, Tensor& label, Tensor& helper, Tensor& weight); 

  // Layer implementations
  virtual bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  virtual bool Connect (const std::vector< CombinedTensor* >& inputs,
                        const std::vector< CombinedTensor* >& outputs);
  virtual void FeedForward() { }
  virtual void BackPropagate() { }
  
private:
  CombinedTensor* data_ = nullptr;
  CombinedTensor* label_ = nullptr;
  CombinedTensor* helper_ = nullptr;
  CombinedTensor* weight_ = nullptr;
};
}

#endif