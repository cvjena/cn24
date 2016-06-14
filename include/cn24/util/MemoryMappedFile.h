/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_MEMORYMAPPEDFILE_H
#define CONV_MEMORYMAPPEDFILE_H

#include <vector>

#include "Tensor.h"

namespace Conv {

class MemoryMappedFile {
public:
  explicit MemoryMappedFile(std::string path);
  ~MemoryMappedFile();

  void* GetAddress() const { return address_; }
  std::size_t GetLength() const { return length_; }

private:
  void* address_ = nullptr;
  std::size_t length_ = 0;

  int file_descriptor = 0;
};

}

#endif
