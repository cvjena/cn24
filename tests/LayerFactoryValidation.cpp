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
  Conv::System::Init();
  
  std::vector<std::string> working_descriptors = {
    "hmax(mu=0.1 weight=1.0)",
    "convolution(size=3x3 pad=3x4)",
    "convolution(size=3x3)",
    "convolution()",
    "convolution"
  };
  std::vector<std::string> nonworking_descriptors = {
    "",
    "convolution(size=3x3 )",
    "convolution(size=3x3)()",
    "convolution(size=4x4, pad=2x2)",
    "?convolution(size=4x4 pad=2x2)",
    "convolution size=4x4 pad=2x2"
  };
  
  bool failed = false;
  
  for(const std::string& descriptor : working_descriptors) {
    if(!Conv::LayerFactory::IsValidDescriptor(descriptor)) {
      LOGERROR << "Valid descriptor rejected: " << descriptor;
      failed = true;
    }
  }
  
  for(const std::string& descriptor : nonworking_descriptors) {
    if(Conv::LayerFactory::IsValidDescriptor(descriptor)) {
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