/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include "TensorRegistry.h"

namespace Conv {
  
void TensorRegistry::RegisterTensor(Tensor* tensor) {
  std::unordered_set<Tensor*>::const_iterator it = tensors_.find(tensor);
  if(it != tensors_.end()) {
    LOGWARN << "Double registered tensor " << tensor << " " << *tensor;
    return;
  }
  
  tensors_.insert(tensor);
}

void TensorRegistry::DeregisterTensor(Tensor* tensor)
{
  std::unordered_set<Tensor*>::const_iterator it = tensors_.find(tensor);
  if(it == tensors_.end()) {
    LOGWARN << "Double deregistered tensor " << tensor << " " << *tensor;
    return;
  }
  
  tensors_.erase(tensor);
}


}