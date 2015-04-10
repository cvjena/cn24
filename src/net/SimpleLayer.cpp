/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <vector>

#include "Log.h"
#include "NetGraph.h"

#include "SimpleLayer.h"

namespace Conv {
  
bool SimpleLayer::Connect (const std::vector< CombinedTensor* >& inputs,
                           const std::vector< CombinedTensor* >& outputs,
                           const NetStatus* net ){
  // A simple layer has exactly 1 input and output
  if(inputs.size() != 1) {
    LOGERROR << "Number of inputs not 1";
    return false;
  }
  if(outputs.size() != 1) { 
    LOGERROR << "Number of outputs not 1";
    return false;
  }
  
  // Check for null pointers
  if(inputs[0] == nullptr || outputs[0] == nullptr) {
    LOGERROR << "Tried to connect to a null pointer!";
    return false;
  }
  
  // Validate nodes before changing anything
  if(!Connect(inputs[0], outputs[0])) {
    LOGERROR << "Nodes failed validation and did not connect!";
    return false;
  }
    
  input_ = inputs[0];
  output_ = outputs[0];
  net_ = net;
  
  return true;
}

void SimpleLayer::CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers) {
	NetGraphBuffer buffer;
	buffer.description = "Output";
	buffers.push_back(buffer);
}


}