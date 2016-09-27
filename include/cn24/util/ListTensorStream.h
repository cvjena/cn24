/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_LISTTENSORSTREAM_H
#define CONV_LISTTENSORSTREAM_H

#include <cstddef>
#include <string>
#include <iostream>
#include <vector>

#include "Log.h"
#include "Config.h"

#include "Tensor.h"
#include "TensorStream.h"
#include "ClassManager.h"

namespace Conv {
	
	struct ListTensorMetadata {
	public:
		ListTensorMetadata(std::string filename, std::size_t width, std::size_t height, std::size_t maps, std::size_t samples) :
			filename(filename), width(width), height(height), maps(maps), samples(samples) {};

		std::string filename;
		std::size_t width = 0, height = 0, maps = 0, samples = 0;
		bool ignore = false;
	};
  
  class ListTensorStream : public TensorStream {
  public:
    explicit ListTensorStream(ClassManager* class_manager) : class_manager_(class_manager) {};
    ~ListTensorStream() {
    }
    
    unsigned int LoadFiles(std::string imagelist_path, std::string images, std::string labellist_path, std::string labels);
		
    // TensorStream implementations
    std::size_t GetWidth(unsigned int index);
    std::size_t GetHeight(unsigned int index);
    std::size_t GetMaps(unsigned int index);
    std::size_t GetSamples(unsigned int index);
    unsigned int GetTensorCount();
    unsigned int LoadFile(std::string path);
    bool CopySample(const unsigned int source_index, const std::size_t source_sample, Tensor& target, const std::size_t target_sample, const bool scale = false);
		
	private:
		std::vector<ListTensorMetadata> tensors_;
		ClassManager* class_manager_ = nullptr;
  };
  
}

#endif