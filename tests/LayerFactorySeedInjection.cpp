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
  
  unsigned int inject_seed = 532;
  
  std::vector<std::string> descriptors = {
    "{\"layer\":{\"type\":\"convolution\", \"size\":[3,3], \"padding\":[3,4]}}",
    "{\"layer\":{\"type\":\"convolution\", \"size\":[3,3], \"padding\":[3,4], \"seed\":0}}",
    "{\"layer\":{\"type\":\"convolution\", \"padding\":[3,4]}}",
    "{\"layer\":{\"type\":\"convolution\"}}",
    "{\"layer\":\"convolution\"}"
  };
  
  bool failed = false;
  
  for(unsigned int s = 0; s < descriptors.size(); s++) {
    Conv::JSON descriptor = Conv::JSON::parse(descriptors[s]);
    Conv::JSON actual_descriptor = Conv::LayerFactory::InjectSeed(descriptor, inject_seed);
		try{
			if((unsigned int)actual_descriptor["layer"]["seed"] != inject_seed) {
				LOGERROR << "Injecting into descriptor " << descriptor << ", expected seed not present. Actual seed: " << actual_descriptor["layer"]["seed"]
				<< ", actual: " << actual_descriptor;
				failed = 1;
			}
		} catch(std::exception e) {
			LOGERROR << "Exception during test, current descriptor: " << descriptor.dump(2);
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
