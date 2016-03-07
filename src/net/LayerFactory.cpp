/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <string>
#include <regex>

#include "LayerFactory.h"

namespace Conv {
Layer* LayerFactory::ConstructLayer(std::string descriptor) {
  Layer* layer = nullptr;
  return layer;
}
}