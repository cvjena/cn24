/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <iostream>
#include <fstream>
#include <string>

#ifdef BUILD_POSIX
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include "TensorStream.h"
#include "FloatTensorStream.h"
#include "CompressedTensorStream.h"
#include "ListTensorStream.h"

namespace Conv {
  
TensorStream* TensorStream::FromFile(std::string path) {
  if(path.compare(0,5,"list:") == 0) {
    LOGDEBUG << "Is list tensor, loading...";
    ListTensorStream* lts = new ListTensorStream();
    lts->LoadFile(path.replace(0,5,""));
    return lts;
  }
  std::ifstream input_stream(path, std::ios::in | std::ios::binary);
  if(!input_stream.good()) {
    FATAL("Cannot open file: " << path);
  }
  uint64_t magic = 0;
  
  input_stream.read((char*)&magic, sizeof(uint64_t)/sizeof(char));
  input_stream.close();
  
  if(magic == CN24_CTS_MAGIC) {
    LOGDEBUG << "Is compressed tensor, loading...";
    CompressedTensorStream* cts = new CompressedTensorStream();
    cts->LoadFile(path);
    return cts;
  } else {
    LOGDEBUG << "Is float tensor, loading...";
    FloatTensorStream* fts = new FloatTensorStream();
    fts->LoadFile(path);
    return fts;
  }
}

}