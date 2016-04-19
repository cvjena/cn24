/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_LISTTENSORSTREAM_H
#define CONV_LISTTENSORSTREAM_H

#include <cstddef>
#include <string>
#include <iostream>

#include "Log.h"
#include "Config.h"

#include "Tensor.h"
#include "TensorStream.h"

namespace Conv {
  
  class ListTensorStream : public TensorStream {
  public:
    
    ~ListTensorStream() {
    }
    
    // TensorStream implementations
    std::size_t GetWidth(unsigned int index);
    std::size_t GetHeight(unsigned int index);
    std::size_t GetMaps(unsigned int index);
    std::size_t GetSamples(unsigned int index);
    unsigned int GetTensorCount();
    unsigned int LoadFile(std::string path);
    bool CopySample(const unsigned int source_index, const std::size_t source_sample, Tensor& target, const std::size_t target_sample);
  };
  
}

#endif