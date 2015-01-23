/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file NonLinearityLayer.h
 * \class NonLinearityLayer
 * \brief This layer introduces a non-linearity (activation function)
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_NONLINEARITYLAYER_H
#define CONV_NONLINEARITYLAYER_H

#include "CombinedTensor.h"
#include "SimpleLayer.h"

namespace Conv {

// This macro is a class declaration for a typical nonlinearity layer
#define NL_LAYER(name) class name##Layer : public NonLinearityLayer {\
public: \
  name##Layer() { LOGDEBUG << "Instance created, nl: " << #name; } \
  void FeedForward(); \
  void BackPropagate(); \
};


#define NL_LAYER_OPENCL(name) class name##Layer : public NonLinearityLayer {\
public: \
  name##Layer() { LOGDEBUG << "Instance created, nl: " << #name; } \
  void FeedForward(); \
  void BackPropagate(); \
  bool IsOpenCLAware() { return true; } \
};

class NonLinearityLayer : public SimpleLayer {
public:
  NonLinearityLayer();
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                              std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  virtual void FeedForward() = 0;
  virtual void BackPropagate() = 0;
};


#ifdef BUILD_OPENCL
NL_LAYER_OPENCL(Tanh)
NL_LAYER_OPENCL(Sigmoid)
#else
NL_LAYER(Tanh)
NL_LAYER(Sigmoid)
#endif
NL_LAYER(ReLU)
NL_LAYER(Softmax)
}

#endif
