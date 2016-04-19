/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ListTensorStream.h"

namespace Conv {
  std::size_t ListTensorStream::GetWidth(unsigned int index) {
    return 0;
  }
  
  std::size_t ListTensorStream::GetHeight(unsigned int index) {
    return 0;
  }
  
  std::size_t ListTensorStream::GetMaps(unsigned int index) {
    return 0;
  }
  
  std::size_t ListTensorStream::GetSamples(unsigned int index) {
    return 0;
  }
  
  unsigned int ListTensorStream::GetTensorCount() {
    return 0;
  }
  
  unsigned int ListTensorStream::LoadFile(std::string path) {
    LOGDEBUG << "Loading file: " << path;
    return 0;
  }
  
  bool ListTensorStream::CopySample(const unsigned int source_index, const std::size_t source_sample, Conv::Tensor &target, const std::size_t target_sample) {
    return false;
  }
}