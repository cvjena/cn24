/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>

#include <vector>
#include <string>

int main(int argc, char* argv[]) {
  UNREFERENCED_PARAMETER(argc);
  UNREFERENCED_PARAMETER(argv);
  Conv::System::Init();
  
  std::vector<std::string> descriptors = {
    "{\"layer\":{\"type\":\"convolution\", \"size\":[3,3], \"padding\":[3,4]}}",
    "{\"layer\":{\"type\":\"convolution\", \"padding\":[3,4]}}",
    "{\"layer\":{\"type\":\"convolution\"}}",
    "{\"layer\":\"convolution\"}"
  };
  std::vector<std::string> expected_layertypes = {
    "convolution",
    "convolution",
    "convolution",
    "convolution"
  };
  
  bool failed = false;
  
  for(unsigned int s = 0; s < descriptors.size(); s++) {
    Conv::JSON descriptor = Conv::JSON::parse(descriptors[s]);
    std::string& excpected_layertype = expected_layertypes[s];
    std::string actual_layertype = Conv::LayerFactory::ExtractLayerType(descriptor);
    if(actual_layertype.compare(excpected_layertype) != 0) {
      LOGERROR << "Extracting descriptor " << descriptor << ", expected: " << excpected_layertype
      << ", actual: " << actual_layertype;
      failed = 1;
    }
  }
  
  LOGEND;
  if(failed) {
    return -1;
  } else {
    return 0;
  }
}