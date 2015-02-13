/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file SpatialPriorLayer.h
 * @class SpatialPriorLayer
 * This class adds two feature maps that contain the normalized pixel
 * coordinates.
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_SPATIALPRIORLAYER_H
#define CONV_SPATIALPRIORLAYER_H

#include "SimpleLayer.h"

namespace Conv {

class SpatialPriorLayer : public SimpleLayer {
public:  
  /**
   * @brief Constructs a SpatialPriorLayer.
   */
  SpatialPriorLayer();
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs, std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
};

}

#endif
