/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Log.h"
#include "TensorViewer.h"

namespace Conv {

TensorViewer::TensorViewer () {
  LOGDEBUG << "Instance created.";
}

void TensorViewer::show ( Tensor* tensor, const std::string& title, bool autoclose, unsigned int map, unsigned int sample ) {
  UNREFERENCED_PARAMETER(autoclose);
  UNREFERENCED_PARAMETER(map);
  UNREFERENCED_PARAMETER(sample);
  LOGWARN << "Cannot show Tensor: " << tensor << ", " << title;
}

}
