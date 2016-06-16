/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifdef BUILD_LIBARCHIVE
extern "C" {
#include <archive.h>
#include <archive_entry.h>
}
#endif

#include <regex>
#include <iomanip>
#include "MemoryMappedTar.h"

namespace Conv {

MemoryMappedTar::MemoryMappedTar(void* address, std::size_t length, MemoryMappedTarFileInfoSink* sink, std::vector<MemoryMappedTarFileInfo>* parent_files) {

#ifdef BUILD_LIBARCHIVE
  archive* tar_archive;

  target_files_ = parent_files == nullptr ? &files_ : parent_files;
  MemoryMappedTarFileInfoSink* target_sink = sink == nullptr ? this : sink;

  tar_archive = archive_read_new();
  archive_read_support_filter_all(tar_archive);
  archive_read_support_format_all(tar_archive);

  if(archive_read_open_memory(tar_archive, address, length) != ARCHIVE_OK) {
    FATAL("Reading archive from memory failed!");
  }

  archive_entry* tar_entry;
  while(archive_read_next_header(tar_archive, &tar_entry) == ARCHIVE_OK) {
    MemoryMappedTarFileInfo info;
    const char* pathname = archive_entry_pathname(tar_entry);
    info.filename = pathname;

    const void* buff = nullptr;
    size_t len = 0;
    off_t offset = 0;
    archive_read_data_block(tar_archive, &buff, &len, &offset);
    if(offset != 0) {
      FATAL("Not handling non-zero offsets right now!");
    }

    info.data = (datum*) buff;
    info.length = len;

    std::regex tar_ending = std::regex(".*\\.tar$", std::regex::extended);
    if(std::regex_match(info.filename, tar_ending)) {
      MemoryMappedTar* inner_mm_tar = new MemoryMappedTar((void*) buff, len, target_sink, target_files_);
      child_tars_.push_back(inner_mm_tar);
    } else {
      sink->Process(info);
    }
  }
  archive_ptr_ = tar_archive;
#else
  LOGERROR << "This build configuration does not support tar files, please check if libarchive is present!";
#endif
}

MemoryMappedTar::~MemoryMappedTar() {
#ifdef BUILD_LIBARCHIVE
  if(archive_ptr_ != nullptr) {
    archive_read_free((archive*)archive_ptr_);
  }
  for(MemoryMappedTar* child_tar : child_tars_) {
    delete child_tar;
  }
#endif
}

void MemoryMappedTar::Process(const MemoryMappedTarFileInfo &file_info) {
  target_files_->push_back(file_info);
}

}
