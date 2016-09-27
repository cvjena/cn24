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
  
  std::vector<std::string> working_descriptors = {
    "{\"layer\":{\"type\":\"convolution\", \"size\":[3,3], \"padding\":[3,4]}}",
    "{\"layer\":{\"type\":\"convolution\", \"padding\":[3,4]}}",
    "{\"layer\":{\"type\":\"convolution\"}}",
    "{\"layer\":\"convolution\"}"
  };
  std::vector<std::string> nonworking_descriptors = {
    "{\"layer\":[]}",
    "{\"layers\":{\"type\":\"convolution\"}}",
    "{\"layer\":{\"sort\":\"convolution\"}}",
    "{\"type\":\"convolution\"}",
		"{}"
  };
  
  bool failed = false;
  
  for(const std::string& descriptor : working_descriptors) {
    if(!Conv::LayerFactory::IsValidDescriptor(Conv::JSON::parse(descriptor))) {
      LOGERROR << "Valid descriptor rejected: " << descriptor;
      failed = true;
    }
  }
  
  for(const std::string& descriptor : nonworking_descriptors) {
    if(Conv::LayerFactory::IsValidDescriptor(Conv::JSON::parse(descriptor))) {
      LOGERROR << "Invalid descriptor accepted: " << descriptor;
      failed = true;
    }
  }
  
  LOGEND;
  if(failed) {
    return -1;
  } else {
    return 0;
  }
}