/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file Layer.h
 * @class Layer
 * @brief Abstract class representing a layer in the neural network.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CONV_LAYER_H
#define CONV_LAYER_H

#include <string>
#include <vector>

#include "Tensor.h"
#include "CombinedTensor.h"

namespace Conv {

  class Net;
  class Trainer;
  class GradientTester;
	class NetGraphBuffer;
class Layer {
  friend class Trainer;
  friend class GradientTester;
public:
  /**
   * @brief Creates a CombinedTensor vector given an input.
   *
   * Note that this does not connect the LayerOutputs to this layer.
   * You need to call Connect for this.
   *
   * @param inputs The inputs to the layer
   * @returns True on success, false for incompatible inputs
   */
  virtual bool CreateOutputs (const std::vector<CombinedTensor*>& inputs,
                              std::vector<CombinedTensor*>& outputs) = 0;

  /**
   * @brief Connects this Layer to the inputs and outputs.
   *
   * @param inputs The inputs to the layer
   * @param outputs The outputs to the layer
   */
  virtual bool Connect (const std::vector<CombinedTensor*>& inputs,
                        const std::vector<CombinedTensor*>& outputs,
                        const Net* net) = 0;
                        
  /**
   * @brief Performs a forward pass
   */
  virtual void FeedForward() = 0;

  /**
   * @brief Performs a backward pass
   */
  virtual void BackPropagate() = 0;

  /**
   * @brief Returns a reference to the parameters.
   */
  inline const std::vector<CombinedTensor*>& parameters() const {
    return parameters_;
  }

  /**
   * @brief Sets the Layer's local learning rate.
   */
  inline void SetLocalLearningRate (const datum local_lr) {
    LOGDEBUG << "Setting local learning rate to " << local_lr;
    local_lr_ = local_lr;
  }

  /**
   * @brief Enables or disables backpropagation.
   */
  inline void SetBackpropagationEnabled (const bool backprop_enabled) {
    backprop_enabled_ = backprop_enabled;
  }

  /**
   * @brief This is called by the net when this layer has a child layer.
   */
  virtual void OnLayerConnect (Layer* next_layer) {
    gain = next_layer->Gain();
  }

  /**
   * @brief Returns the layer's gain
   */
  virtual unsigned int Gain() { return gain; }

  /**
   * @brief Returns true if the layer is OpenCL aware
   * 
   * If this returns false, the net will make sure that any
   * input and output tensors are in the CPU's memory and not
   * the GPU's.
   */
  virtual bool IsOpenCLAware() { return false; }

	virtual std::string GetLayerDescription() { return "Layer"; }
	virtual void CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers) {}
protected:
  /**
   * @brief These CombinedTensors contain the weights and biases.
   *
   * The training routine will update these
   */
  std::vector<CombinedTensor*> parameters_;

  /**
   * @brief This factor sets the local learning rate
   */
  datum local_lr_ = 1.0;

  /**
   * @brief This boolean enables backpropagation.
   */
  bool backprop_enabled_ = true;

  unsigned int gain = 0;
};

}

#endif
