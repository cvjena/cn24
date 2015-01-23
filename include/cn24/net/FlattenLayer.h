/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file FlattenLayer.h
 * \class FlattenLayer
 * \brief Layer that shadows a Tensor in a different shape.
 * 
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_FLATTENLAYER_H
#define CONV_FLATTENLAYER_H

#include "SimpleLayer.h"

namespace Conv {

class FlattenLayer : public SimpleLayer {
public:  
  FlattenLayer();
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
#ifdef BUILD_OPENCL
  bool IsOpenCLAware() { return true; }
#endif
};

}

#endif