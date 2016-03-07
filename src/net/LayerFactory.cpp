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
bool LayerFactory::IsValidDescriptor(std::string descriptor) {
  bool valid = std::regex_match(descriptor, std::regex("^[a-z]+(\\("
    "("
    "[a-z]+=[a-zA-Z0-9]+"
    "( [a-z]+=[a-zA-Z0-9]+)*"
    ")?"
    "\\))?$",std::regex::extended));
  return valid;
}
  
std::string LayerFactory::ExtractConfiguration(std::string descriptor) {
  std::smatch config_match;
  bool has_nonempty_configuration = std::regex_match(descriptor, config_match, std::regex("[a-z]+\\((.+)\\)",std::regex::extended));
  if(has_nonempty_configuration && config_match.size() == 2) {
    return config_match[1];
  } else {
    return "";
  }
}
  
std::string LayerFactory::ExtractLayerType(std::string descriptor) {
  std::smatch config_match;
  bool has_layertype = std::regex_match(descriptor, config_match, std::regex("([a-z]+)(\\(.*\\))?",std::regex::extended));
  if(has_layertype && config_match.size() > 1) {
    return config_match[1];
  } else {
    return "";
  }
}
  
Layer* LayerFactory::ConstructLayer(std::string descriptor) {
  if (!IsValidDescriptor(descriptor))
    return nullptr;
  std::string configuration = ExtractConfiguration(descriptor);
  
  Layer* layer = nullptr;
  return layer;
}
  
}