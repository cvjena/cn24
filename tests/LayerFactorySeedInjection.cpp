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
  
  unsigned int inject_seed = 532;
  
  std::vector<std::string> descriptors = {
    "convolution(size=3x3 pad=3x4)",
    "convolution(size=3x3 pad=3x4 seed=0)",
    "convolution(size=3x3 seed=0 pad=3x4)",
    "convolution(size=3x3)",
    "convolution()",
    "convolution"
  };
  std::vector<std::string> expected_descriptors = {
    "convolution(size=3x3 pad=3x4 seed=532)",
    "convolution(size=3x3 pad=3x4 seed=532)",
    "convolution(size=3x3 seed=532 pad=3x4)",
    "convolution(size=3x3 seed=532)",
    "convolution(seed=532)",
    "convolution(seed=532)"
  };
  
  bool failed = false;
  
  for(unsigned int s = 0; s < descriptors.size(); s++) {
    std::string& descriptor = descriptors[s];
    std::string& expected_descriptor = expected_descriptors[s];
    std::string actual_descriptor = Conv::LayerFactory::InjectSeed(descriptor, inject_seed);
    if(actual_descriptor.compare(expected_descriptor) != 0) {
      LOGERROR << "Injecting into descriptor " << descriptor << ", expected: " << expected_descriptor
      << ", actual: " << actual_descriptor;
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