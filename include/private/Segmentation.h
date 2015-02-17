/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file Segmentation.h
 * @class Segmentation
 * @brief Utility functions for segmentation.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_SEGMENTATION_H
#define CONV_SEGMENTATION_H

#include "Tensor.h"

namespace Conv {

class Segmentation {
public:
  static void ExtractPatches (const int patchsize_x,
                              const int patchsize_y,
                              Tensor& target, Tensor& helper,
                              const Tensor& source,
                              const int source_sample = 0,
                              bool substract_mean = false
                             );
  
  static void ExtractLabels (const int patchsize_x,
			     const int patchsize_y,
			     Tensor& labels, Tensor& weight,
			     const Tensor& source,
			     const int source_sample = 0,
			     const int ignore_class = -1
			    );
};

}

#endif
