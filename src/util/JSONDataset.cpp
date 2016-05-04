/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include "JSONParsing.h"

#include "Dataset.h"
#include "TensorStream.h"

namespace Conv {

	
JSONDataset::JSONDataset() {
	
}

JSONDataset::~JSONDataset() {
	
}

void JSONDataset::LoadFile(std::istream& file, bool dont_load, DatasetLoadSelection selection) {
	// TODO Validate JSON
	
	JSON dataset_json = JSON::parse(file);
	std::string dataset_name = dataset_json["name"];
	
	LOGINFO << "Loading dataset \"" << dataset_name << "\"";
	
	// Load metadata
	if(is_first_dataset) {
		// Set error function
		error_function_ = DefaultLocalizedErrorFunction;
		
		// Count classes
		unsigned int class_count = dataset_json["classes"].size();
		for(unsigned int c = 0; c < class_count; c++) {
			JSON class_json = dataset_json["classes"][c];
			
			std::string class_name = class_json["name"];
			unsigned int class_color = 0;
			datum class_weight = 1.0;
			
			if(class_json.count("color") == 1) {
				std::string class_color_string = class_json["color"];
				unsigned long color_val_l = std::strtoul (class_color_string.c_str(), nullptr, 16);
				class_color = (unsigned int) color_val_l;
			}
			
			if(class_json.count("weight") == 1) {
				class_weight = class_json["weight"];
			}
			
			class_names_.push_back(class_name);
			class_weights_.push_back(class_weight);
			class_colors_.push_back(class_color);
		}
		classes_ = class_count;
	} else {
		// TODO Validate similarity
	}
	
	// Load data
	unsigned int data_elements = dataset_json["data"].size();
	for(unsigned int d = 0; d < data_elements; d++) {
		JSON element_json = dataset_json["data"][d];
		std::string element_type = element_json["type"];
		if(element_type.compare("tensor_stream") == 0) {
			std::string segment = element_json["segment"];
			std::string filename = element_json["filename"];
			TensorStream* tensor_stream = TensorStream::FromFile(filename, class_colors_);
			unsigned int tensor_count = tensor_stream->GetTensorCount();
			TensorStreamAccessor accessor;
			
			// Iterate through tensors to determine max dimensions
			for (unsigned int t = 0; t < (tensor_count_training_ / 2); t++) {
				if(tensor_stream->GetWidth(2*t) > max_width_)
					max_width_ = tensor_stream->GetWidth(2*t);
				
				if(tensor_stream->GetHeight(2*t) > max_height_)
					max_height_ = tensor_stream->GetHeight(2*t);
			}
			
			if(tensor_count == 0) {
				LOGWARN << "Empty tensor stream for segment \"" << segment << "\", filename: " << filename;
			}
		}
	}
	
	// Increase max_width and max_height for better pooling
	if (max_width_ & 1)
    max_width_++;
  if (max_height_ & 1)
    max_height_++;
  
  if (max_width_ & 2)
    max_width_+=2;
  if (max_height_ & 2)
    max_height_+=2;

  if (max_width_ & 4)
    max_width_+=4;
  if (max_height_ & 4)
    max_height_+=4;
  
  if (max_width_ & 8)
    max_width_+=8;
  if (max_height_ & 8)
    max_height_+=8;
  
  if (max_width_ & 16)
    max_width_+=16;
  if (max_height_ & 16)
    max_height_+=16;
  
  if (max_width_ & 32)
    max_width_+=32;
  if (max_height_ & 32)
    max_height_+=32;
	
}	


bool JSONDataset::GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
	return false;
}

bool JSONDataset::GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& helper_tensor, Tensor& weight_tensor,  unsigned int sample, unsigned int index) {
	return false;
}

}