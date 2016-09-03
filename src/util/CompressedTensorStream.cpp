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

#include "CompressedTensorStream.h"

namespace Conv {
  
unsigned int CompressedTensorStream::LoadFile(std::string path)
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

  uint64_t magic = 0;
  input_stream.read((char*)&magic, sizeof(uint64_t)/sizeof(char));
  
  if(magic != CN24_CTS_MAGIC) {
    FATAL("Wrong magic at start of stream!");
  }

  // Go through file
  std::cout << std::endl << std::flush;
  
  while (!input_stream.eof()) {
    CompressedTensor* tensor = new CompressedTensor();
#ifdef BUILD_POSIX
    tensor->Deserialize (input_stream, false, true, input_fd);
#else
    tensor->Deserialize (input_stream, false);
#endif

    if (tensor->elements() == 0)
      break;
    
    if(tensor->elements() > max_elements_)
      max_elements_ = tensor->elements();

    tensors_.push_back(tensor);
    std::cout << "." << std::flush;
    input_stream.peek();
  }
  
  temp_tensor_.Resize(1,max_elements_);
  return 0;
}

bool CompressedTensorStream::CopySample(const unsigned int source, const std::size_t source_sample,
                                   Conv::Tensor& target, const std::size_t target_sample, const bool scale)
{
  if(source < tensors_.size()) {
    CompressedTensor* const ctensor = tensors_[source];
    if(source_sample == 0 && ctensor->width() == target.width() && ctensor->height() == target.height() && ctensor->maps() == target.maps() && ctensor->samples() == 1) {
      // This is a little hack for faster loading of certain datasets
#ifdef BUILD_OPENCL
      target.MoveToCPU();
#endif
      datum* old_data_ptr = temp_tensor_.data_ptr();
      datum* direct_ptr = target.data_ptr(0, 0, 0, target_sample);
      temp_tensor_.Resize(1, max_elements_, 1, 1, direct_ptr, false, true);
      ctensor->Decompress(temp_tensor_, temp_tensor_.data_ptr());
      
      temp_tensor_.Resize(1, max_elements_, 1, 1, old_data_ptr, false, true);
      return true;
    } else {
      ctensor->Decompress(temp_tensor_, temp_tensor_.data_ptr());
      return Tensor::CopySample(temp_tensor_, source_sample, target, target_sample, false, scale);
    }
  } else
    return false;
}

}





