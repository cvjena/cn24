/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file InputLayer.h
 * @class InputLayer
 * @brief A simple layer that always outputs the same Tensor.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
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
  /**
	* @brief Creates an InputLayer from a Tensor
	*
	* @param data The Tensor to output to subsequent layers
	*/
  explicit InputLayer(Tensor& data);

  /**
	* @brief Creates an InputLayer from a data and a helper Tensor
	*
	* @param data The data Tensor to output to subsequent layers
	* @param helper The helper Tensor to output to subsequent layers
	*/
  InputLayer(Tensor& data, Tensor& helper); 

  /**
	* @brief Creates an InputLayer from data, helper, label and weight Tensors
	*
	* @param data The data Tensor to output to subsequent layers
	* @param label The label Tensor to output to subsequent layers
	* @param helper The helper Tensor to output to subsequent layers
	* @param weight The weight Tensor to output to subsequent layers
	*/
  InputLayer(Tensor& data, Tensor& label, Tensor& helper, Tensor& weight); 

  // Layer implementations
  virtual bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  virtual bool Connect (const std::vector< CombinedTensor* >& inputs,
                        const std::vector< CombinedTensor* >& outputs,
                        const NetStatus* net );
  virtual void FeedForward() { }
  virtual void BackPropagate() { }
  
	std::string GetLayerDescription() { return "Simple Input Layer"; }
	void CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers);
private:
  CombinedTensor* data_ = nullptr;
  CombinedTensor* label_ = nullptr;
  CombinedTensor* helper_ = nullptr;
  CombinedTensor* weight_ = nullptr;
};
}

#endif
