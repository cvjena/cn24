/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

/**
 * @file TensorViewer.h
 * @class TensorViewer
 * @brief A GTK+ interface to view Tensors.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CN24_TENSORVIEWER_H
#define CN24_TENSORVIEWER_H

#include "Config.h"
#include <string>

namespace Conv {
class Tensor;
class TensorViewer {
public:
  TensorViewer();
  void show(Tensor* tensor, const std::string& title = "Tensor Viewer",bool autoclose = false, unsigned int map = 0, unsigned int sample = 0);
};

}
#endif
