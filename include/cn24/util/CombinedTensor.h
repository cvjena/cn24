/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file LayerOutput.h
 * @class LayerOutput
 * @brief Combination of Tensors to represent errors and outputs of Layers.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_COMBINEDTENSOR_H
#define CONV_COMBINEDTENSOR_H

#include <cstddef>

#include "Tensor.h"

namespace Conv {

class CombinedTensor {
public:
  /**
   * @brief Constructs a CombinedTensor with two Tensors of the specified size.
   * 
   * This is useful for storing data and gradients or differences in the same
   * place.
   * 
   * Contract:
   * Code can expect the delta and data Tensors to have the same shape.
   * Code must _not_ reshape only one of the Tensors.
   *
   * @see Tensor.h for size parameter documentation
   */
  explicit CombinedTensor (const std::size_t samples,
                           const std::size_t width = 1,
                           const std::size_t height = 1,
                           const std::size_t maps = 1,
                           DatasetMetadataPointer* metadata = nullptr,
                           bool is_dynamic = false) :
    data (samples, width, height, maps),
    delta (samples, width, height, maps), metadata(metadata), is_dynamic(is_dynamic) {}


  Tensor data;
  Tensor delta;
  DatasetMetadataPointer* metadata;
  bool is_dynamic;
};

}

#endif
