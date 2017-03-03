/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#ifndef CONV_TENSORREGISTRY_H
#define CONV_TENSORREGISTRY_H

#include "Tensor.h"
#include <unordered_set>

namespace Conv {

class TensorRegistry {
public:
  void RegisterTensor(Tensor* tensor);
  void DeregisterTensor(Tensor* tensor);
  
  typedef std::unordered_set<Tensor*>::const_iterator const_iterator;
  
  const_iterator begin() { return tensors_.begin(); }
  const_iterator end() { return tensors_.end(); }
  std::unordered_set<Tensor*>::size_type size() const { return tensors_.size(); }
  
private:
  std::unordered_set<Tensor*> tensors_;
};

}

#endif