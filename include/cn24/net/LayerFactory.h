
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
  static bool IsValidDescriptor(std::string descriptor);
  static std::string ExtractConfiguration(std::string descriptor);
  static std::string ExtractLayerType(std::string descriptor);
  static std::string InjectSeed(std::string descriptor, unsigned int seed);
	
	// New JSON methods
  static Layer* ConstructLayer(JSON descriptor);
  static bool IsValidDescriptor(JSON descriptor);
  static JSON ExtractConfiguration(JSON descriptor);
  static std::string ExtractLayerType(JSON descriptor);
	static JSON InjectSeed(JSON descriptor, unsigned int seed);
  private:
  void descript();
};
  
}

#endif
