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
  const char* filename;
  std::size_t length;
  datum* data;
};

class MemoryMappedTarFileInfoSink {
public:
  virtual void Process(const MemoryMappedTarFileInfo& file_info) = 0;
};

class MemoryMappedTar : public MemoryMappedTarFileInfoSink {
public:
  explicit MemoryMappedTar(void* address, std::size_t length, MemoryMappedTarFileInfoSink* sink = nullptr, std::vector<MemoryMappedTarFileInfo>* parent_files = nullptr);
  ~MemoryMappedTar();

  unsigned int GetFileCount() const { return files_.size(); };
  const MemoryMappedTarFileInfo& GetFileInfo(unsigned int index) const { return files_[index]; };

  virtual void Process(const MemoryMappedTarFileInfo& file_info);

private:
  std::vector<MemoryMappedTarFileInfo> files_;
  std::vector<MemoryMappedTar*> child_tars_;
  void* archive_ptr_ = nullptr;

  std::vector<MemoryMappedTarFileInfo>* target_files_ = nullptr;
};

}

#endif