/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <string>
#include <sstream>
#include <regex>

#include "ConvolutionLayer.h"
#include "NonLinearityLayer.h"
#include "AdvancedMaxPoolingLayer.h"

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
  
#define CONV_LAYER_TYPE(ltype,lclass) else if (layertype.compare(ltype) == 0) { \
layer = new lclass (configuration) ; \
}
  
Layer* LayerFactory::ConstructLayer(std::string descriptor) {
  if (!IsValidDescriptor(descriptor))
    return nullptr;
  std::string configuration = ExtractConfiguration(descriptor);
  std::string layertype = ExtractLayerType(descriptor);
  
  Layer* layer = nullptr;
  if(layertype.length() == 0) {
    // Leave layer a nullptr
  }
  CONV_LAYER_TYPE("convolution", ConvolutionLayer)
  CONV_LAYER_TYPE("amaxpooling", AdvancedMaxPoolingLayer)
  CONV_LAYER_TYPE("tanh", TanhLayer)
  CONV_LAYER_TYPE("sigm", SigmoidLayer)
  CONV_LAYER_TYPE("relu", ReLULayer)
  
  return layer;
}
  
std::string LayerFactory::InjectSeed(std::string descriptor, unsigned int seed) {
  if(IsValidDescriptor(descriptor)) {
    std::string configuration = ExtractConfiguration(descriptor);
    std::string layertype = ExtractLayerType(descriptor);
    
    std::stringstream seed_ss;
    seed_ss << "seed=" << seed;
    
    bool already_has_seed = std::regex_match(configuration, std::regex(".*seed=[0-9]+.*", std::regex::extended));
    if(already_has_seed) {
      std::string new_descriptor = std::regex_replace(descriptor, std::regex("seed=([0-9])+", std::regex::extended), seed_ss.str());
      return new_descriptor;
    } else {
      std::stringstream new_descriptor_ss;
      new_descriptor_ss << layertype << "(";
      if(configuration.length() > 0) {
        new_descriptor_ss << configuration << " ";
      }
      new_descriptor_ss << seed_ss.str() << ")";
      std::string new_descriptor = new_descriptor_ss.str();
      return new_descriptor;
    }
  } else {
    return descriptor;
  }
}
  
}