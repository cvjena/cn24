/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Log.h"


#include <fstream>

#ifdef BUILD_POSIX
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <errno.h>
#include <unistd.h>
#endif

#include "MemoryMappedFile.h"

namespace Conv {

MemoryMappedFile::MemoryMappedFile(std::string path) {

#ifdef BUILD_POSIX
  // Get file's length
  std::ifstream input_stream(path, std::ios::binary | std::ios::ate);
  if(!input_stream.good()) {
    FATAL("Cannot open file: " << path);
  }
  length_ = input_stream.tellg();
  input_stream.close();

  // Get file descriptor for mapping
  int input_fd = open(path.c_str(), O_RDONLY);
  if(input_fd < 0) {
    FATAL("Cannot open file: " << path);
  }
  file_descriptor = input_fd;

#if defined(BUILD_LINUX)
  void* target_mmap = mmap64(nullptr, length_, PROT_READ, MAP_PRIVATE, input_fd, 0);
#elif defined(BUILD_OSX)
  void* target_mmap = mmap(nullptr, length_, PROT_READ, MAP_PRIVATE, input_fd, 0);
#endif

  if(target_mmap == MAP_FAILED) {
    LOGERROR << "Memory map failed: " << errno;
  }
  address_ = target_mmap;

#else
  LOGERROR << "This platform does not support memory mapped files!";
#endif
}

MemoryMappedFile::~MemoryMappedFile() {
#ifdef BUILD_POSIX
  munmap(address_, length_);
  close(file_descriptor);
#endif
}

}
