/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#ifndef CONV_COMPRESSEDTENSORSTREAM_H
#define CONV_COMPRESSEDTENSORSTREAM_H

#include <cstddef>
#include <string>
#include <iostream>

#include "Log.h"
#include "Config.h"

#include "Tensor.h"
#include "CompressedTensor.h"

#include "TensorStream.h"

#define CN24_CTS_MAGIC 0xC24CC24CC24CC24C

namespace Conv {
  
class CompressedTensorStream : public TensorStream {
public: 
  
  ~CompressedTensorStream() {
    for(CompressedTensor* tensor: tensors_) {
      delete tensor;
    }
  }
  
  // TensorStream implementations
  std::size_t GetWidth(unsigned int index) { return index < tensors_.size() ? tensors_[index]->width() : 0; }
  std::size_t GetHeight(unsigned int index) { return index < tensors_.size() ? tensors_[index]->height() : 0; }
  std::size_t GetMaps(unsigned int index) { return index < tensors_.size() ? tensors_[index]->maps() : 0; }
  std::size_t GetSamples(unsigned int index) { return index < tensors_.size() ? tensors_[index]->samples() : 0; }
  unsigned int GetTensorCount() { return tensors_.size(); }
  unsigned int LoadFile(std::string path);
  bool CopySample(const unsigned int source_index, const std::size_t source_sample, Tensor& target, const std::size_t target_sample, const bool scale = false);
private:
  std::vector<CompressedTensor*> tensors_;
  std::size_t max_elements_ = 0;
  Tensor temp_tensor_;
};

}

#endif