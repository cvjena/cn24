/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file SimpleLayer.h
 * @class SimpleLayer
 * @brief Abstract class representing a simple layer (1 input, 1 output).
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CONV_SIMPLELAYER_H
#define CONV_SIMPLELAYER_H

#include <vector>
#include <string>

#include "Layer.h"
#include "CombinedTensor.h"
#include "Net.h"

namespace Conv {
  
class SimpleLayer : public Layer {
public:
  bool Connect(const std::vector<CombinedTensor*>& inputs,
               const std::vector<CombinedTensor*>& outputs,
               const Net* net );
  
  /**
   * @brief Connect the Layer to the CombinedTensors
   * 
   * SimpleLayer will set the output_ and input_ variables only if this
   * returns true.
   * 
   * @param input Input to validate
   * @param output Output to validate
   * @returns True if input and output nodes are correct
   */
  virtual bool Connect(const CombinedTensor* input, CombinedTensor* output) = 0;
	// virtual std::string GetLayerDescription() { return "SimpleLayer"; }

	virtual void CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers);
protected:
  const Net* net_ = nullptr;
  CombinedTensor* input_ = nullptr;
  CombinedTensor* output_ = nullptr;
};

}

#endif
