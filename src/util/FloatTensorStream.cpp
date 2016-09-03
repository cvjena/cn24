/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <iostream>
#include <fstream>

#ifdef BUILD_POSIX
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include "FloatTensorStream.h"

namespace Conv {
  
unsigned int FloatTensorStream::LoadFile(std::string path)
{
  std::ifstream input_stream(path, std::ios::binary | std::ios::in);
  if(!input_stream.good()) {
    FATAL("Cannot open file: " << path);
  }
#ifdef BUILD_POSIX
  int input_fd = open(path.c_str(), O_RDONLY);
  if(input_fd < 0) {
    FATAL("Cannot open file: " << path);
  }
#endif

  // Go through file
  std::cout << std::endl << std::flush;
  
  while (!input_stream.eof()) {
    Tensor* tensor = new Tensor();
#ifdef BUILD_POSIX
    tensor->Deserialize (input_stream, false, true, input_fd);
#else
    tensor->Deserialize (input_stream, false);
#endif

    if (tensor->elements() == 0)
      break;

    tensors_.push_back(tensor);
    std::cout << "." << std::flush;
    input_stream.peek();
  }
  return 0;
}

bool FloatTensorStream::CopySample(const unsigned int source, const std::size_t source_sample,
                                   Conv::Tensor& target, const std::size_t target_sample, const bool scale)
{
  if(source < tensors_.size()) {
    return Tensor::CopySample(*tensors_[source], source_sample, target, target_sample, false, scale);
  } else
    return false;
}

}

