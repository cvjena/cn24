/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include "JSONParsing.h"

#include "Dataset.h"
#include "TensorStream.h"
#include "ListTensorStream.h"

namespace Conv {

	
JSONSegmentationDataset::JSONSegmentationDataset(ClassManager* class_manager) : Dataset(class_manager) {
	
}

JSONSegmentationDataset::~JSONSegmentationDataset() {
	
}

void JSONSegmentationDataset::Load(JSON dataset_json, bool dont_load, DatasetLoadSelection selection) {
  // TODO Make these work
  UNREFERENCED_PARAMETER(dont_load);
  UNREFERENCED_PARAMETER(selection);

	// TODO Validate JSON
	
	std::string dataset_name = dataset_json["name"];
  name_ = dataset_name;

	LOGINFO << "Loading dataset \"" << dataset_name << "\"";
	
	// Load metadata
	if(is_first_dataset) {
    // TODO Enable better error function
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

      bool result = class_manager_->RegisterClassByName(class_name, class_color, class_weight);
      if(!result) {
        FATAL("Could not register class " << class_name << " with ClassManager");
      }
		}
	} else {
		// TODO Validate similarity
	}
	
	// Load data
	unsigned int data_elements = dataset_json["data"].size();
	for(unsigned int d = 0; d < data_elements; d++) {
		JSON element_json = dataset_json["data"][d];
		std::string element_type = element_json["type"];
		if(element_type.compare("tensor_stream") == 0 || element_type.compare("list") == 0) {
      TensorStream* tensor_stream = nullptr;
      std::string filename = "";
			std::string segment = element_json["segment"];

      if(element_type.compare("tensor_stream") == 0) {
        filename = element_json["filename"];
        tensor_stream = TensorStream::FromFile(filename, class_manager_);
      } else if(element_type.compare("list") == 0) {
        filename = "LISTTENSORSTREAM";
        std::string imagelist_path = element_json["imagelist"];
        std::string labellist_path = element_json["labellist"];
        std::string images = element_json["imagepath"];
        std::string labels = element_json["labelpath"];
        tensor_stream = new ListTensorStream(class_manager_);
        dynamic_cast<ListTensorStream*>(tensor_stream)->LoadFiles(imagelist_path, images, labellist_path, labels);
      }

			unsigned int tensor_count = tensor_stream->GetTensorCount();
			
      // Validate tensor count
			if(tensor_count == 0) {
				LOGWARN << "Empty tensor stream for segment \"" << segment << "\", filename: " << filename;
			}

      if(tensor_count % 2 == 1) {
        FATAL("Wrong count (not divisible by 2) in tensor stream for segment \"" << segment << "\", filename: " << filename);
      }

			// Iterate through tensors to determine max dimensions
			for (unsigned int t = 0; t < tensor_count; t++) {
				if(tensor_stream->GetWidth(t) > max_width_)
					max_width_ = tensor_stream->GetWidth(t);
				
				if(tensor_stream->GetHeight(t) > max_height_)
					max_height_ = tensor_stream->GetHeight(t);

        TensorStreamAccessor accessor;
        accessor.tensor_stream = tensor_stream;
        accessor.sample_in_stream = t;

        if(segment.compare("training") == 0) {
          training_accessors_.push_back(accessor);
          tensor_count_training_++;
        } else if(segment.compare("testing") == 0) {
          testing_accessors_.push_back(accessor);
          tensor_count_testing_++;
        } else {
          FATAL("Unknown segment \"" << segment << "\", filename: " << filename);
        }
			}
			
      // Check if input and label map count needs to be set
      if(input_maps_ == 0 && label_maps_ == 0) {
        if(tensor_stream->GetTensorCount() > 0) {
          input_maps_ = tensor_stream->GetMaps(0);
          label_maps_ = tensor_stream->GetMaps(1);
        }
      } else {
        if(tensor_stream->GetTensorCount() > 0) {
          if(input_maps_ != tensor_stream->GetMaps(0) ||
             label_maps_ != tensor_stream->GetMaps(1)) {
            FATAL("Map count mismatch for segment \"" << segment << "\", filename: " << filename);
          }
        }
      }

      // Add TensorStream to cleanup list
      tensor_streams_.push_back(tensor_stream);
		}
	}

	if(dataset_json.count("width") == 1 && dataset_json["width"].is_number() && dataset_json.count("height") == 1 && dataset_json["height"].is_number()) {
    max_width_ = dataset_json["width"];
    max_height_ = dataset_json["height"];

    LOGDEBUG << "Dataset specifies fixed size of " << max_width_ << "x" << max_height_;
  } else {
    // Increase max_width and max_height for better pooling
    if (max_width_ & 1)
      max_width_++;
    if (max_height_ & 1)
      max_height_++;

    if (max_width_ & 2)
      max_width_ += 2;
    if (max_height_ & 2)
      max_height_ += 2;

    if (max_width_ & 4)
      max_width_ += 4;
    if (max_height_ & 4)
      max_height_ += 4;

    if (max_width_ & 8)
      max_width_ += 8;
    if (max_height_ & 8)
      max_height_ += 8;

    if (max_width_ & 16)
      max_width_ += 16;
    if (max_height_ & 16)
      max_height_ += 16;

    if (max_width_ & 32)
      max_width_ += 32;
    if (max_height_ & 32)
      max_height_ += 32;

  }
}


bool JSONSegmentationDataset::GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
  if (index < tensor_count_training_ / 2) {
    bool success = true;

    TensorStream* training_stream = training_accessors_[2 * index].tensor_stream;
    unsigned int index_image = training_accessors_[2 * index].sample_in_stream;
    unsigned int index_label = training_accessors_[2 * index + 1].sample_in_stream;

    success &= training_stream->CopySample(index_image, 0, data_tensor, sample);
    success &= training_stream->CopySample(index_label, 0, label_tensor, sample);

    unsigned int data_width = training_stream->GetWidth(index_image);
    unsigned int data_height = training_stream->GetHeight(index_image);
    
		// Write spatial prior data to helper tensor
		for (unsigned int y = 0; y < data_height; y++) {
			for (unsigned int x = 0; x < data_width; x++) {
				*helper_tensor.data_ptr(x, y, 0, sample) = ((datum)x) / ((datum)data_width - 1);
				*helper_tensor.data_ptr(x, y, 1, sample) = ((datum)y) / ((datum)data_height - 1);
			}
			for (unsigned int x = data_width; x < GetWidth(); x++) {
				*helper_tensor.data_ptr(x, y, 0, sample) = 0;
				*helper_tensor.data_ptr(x, y, 1, sample) = 0;
			}
		}
		for (unsigned int y = data_height; y < GetHeight(); y++) {
			for (unsigned int x = 0; x < GetWidth(); x++) {
				*helper_tensor.data_ptr(x, y, 0, sample) = 0;
				*helper_tensor.data_ptr(x, y, 1, sample) = 0;
			}
		}

    weight_tensor.Clear (0.0, sample);

    // TODO Change boundaries for classification
    #pragma omp parallel for default(shared)
    for (unsigned int y = 0; y < data_height; y++) {
      for (unsigned int x = 0; x < data_width; x++) {
        // TODO Read weights from ClassManager
        const datum class_weight = 1.0; // class_weights_[label_tensor.PixelMaximum(x, y, sample)];
        *weight_tensor.data_ptr (x, y, 0, sample) = error_function_ (x, y, data_width, data_height) * class_weight;
      }
    }

    return success;
  } else return false;

}

bool JSONSegmentationDataset::GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& helper_tensor, Tensor& weight_tensor,  unsigned int sample, unsigned int index) {
  if (index < tensor_count_testing_ / 2) {
    bool success = true;

    TensorStream* testing_stream = testing_accessors_[2 * index].tensor_stream;
    unsigned int index_image = testing_accessors_[2 * index].sample_in_stream;
    unsigned int index_label = testing_accessors_[2 * index + 1].sample_in_stream;

    success &= testing_stream->CopySample(index_image, 0, data_tensor, sample);
    if(!success){LOGDEBUG << "Cannot load sample index " << index << ", image in stream: " << index_image;}
    success &= testing_stream->CopySample(index_label, 0, label_tensor, sample);
    if(!success){LOGDEBUG << "Cannot load sample index " << index << ", label in stream: " << index_label;}
    
    unsigned int data_width = testing_stream->GetWidth(index_image);
    unsigned int data_height = testing_stream->GetHeight(index_image);

		// Write spatial prior data to helper tensor
		for (unsigned int y = 0; y < data_height; y++) {
			for (unsigned int x = 0; x < data_width; x++) {
				*helper_tensor.data_ptr(x, y, 0, sample) = ((datum)x) / ((datum)data_width - 1);
				*helper_tensor.data_ptr(x, y, 1, sample) = ((datum)y) / ((datum)data_height - 1);
			}
			for (unsigned int x = data_width; x < GetWidth(); x++) {
				*helper_tensor.data_ptr(x, y, 0, sample) = 0;
				*helper_tensor.data_ptr(x, y, 1, sample) = 0;
			}
		}
		for (unsigned int y = data_height; y < GetHeight(); y++) {
			for (unsigned int x = 0; x < GetWidth(); x++) {
				*helper_tensor.data_ptr(x, y, 0, sample) = 0;
				*helper_tensor.data_ptr(x, y, 1, sample) = 0;
			}
		}

    weight_tensor.Clear (0.0, sample);

    // TODO Change boundaries for classification
    #pragma omp parallel for default(shared)
    for (unsigned int y = 0; y < data_height; y++) {
      for (unsigned int x = 0; x < data_width; x++) {
        // TODO Read weights from ClassManager
        const datum class_weight = 1.0; // class_weights_[label_tensor.PixelMaximum(x, y, sample)];
        *weight_tensor.data_ptr (x, y, 0, sample) = error_function_ (x, y, data_width, data_height) * class_weight;
      }
    }

    return success;
  } else return false;


}

}
