/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#ifndef CONV_TENSORSTREAM_H
#define CONV_TENSORSTREAM_H

#include <cstddef>
#include <string>
#include <iostream>
#include <vector>

#include "Log.h"
#include "Config.h"
#include "Tensor.h"
#include "ClassManager.h"

namespace Conv {

class TensorStream {
public:
  virtual std::size_t GetWidth(unsigned int index) = 0;
  virtual std::size_t GetHeight(unsigned int index) = 0;
  virtual std::size_t GetMaps(unsigned int index) = 0;
  virtual std::size_t GetSamples(unsigned int index) = 0;
  virtual unsigned int LoadFile(std::string path) = 0;
  
  virtual bool CopySample(const unsigned int source, const std::size_t source_sample,
                          Tensor& target, const std::size_t target_sample, bool scale = false) = 0;
  
  virtual unsigned int GetTensorCount() = 0;
  
  static TensorStream* FromFile(std::string path, ClassManager* class_manager);
};

}

#endif





















