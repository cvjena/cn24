
/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file LayerFactory.h
 * @class LayerFactory
 * @brief Creates layers from configuration strings
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_LAYERFACTORY_H
#define CONV_LAYERFACTORY_H

#include <string>

#include "Layer.h"

namespace Conv {

class LayerFactory {
public:
  static Layer* ConstructLayer(std::string descriptor);
};
  
}

#endif
