/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_MEMORYMAPPEDTAR_H
#define CONV_MEMORYMAPPEDTAR_H

#include <vector>

#include "Tensor.h"

namespace Conv {

struct MemoryMappedTarFileInfo {
public:
  std::string filename;
  std::size_t length;
  datum* data;
};

class MemoryMappedTar {
public:
  explicit MemoryMappedTar(void* address, std::size_t length, std::vector<MemoryMappedTarFileInfo>* parent_files = nullptr);
  ~MemoryMappedTar();

  unsigned int GetFileCount() const { return files_.size(); };
  const MemoryMappedTarFileInfo& GetFileInfo(unsigned int index) const { return files_[index]; };

private:
  std::vector<MemoryMappedTarFileInfo> files_;
  std::vector<MemoryMappedTar*> child_tars_;
  void* archive_ptr_ = nullptr;
};

}

#endif