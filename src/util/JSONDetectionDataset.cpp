/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <fstream>
#include <random>
#include <algorithm>

#include "JSONParsing.h"

#include "Dataset.h"
#include "TensorStream.h"
#include "ListTensorStream.h"

namespace Conv {

	
JSONDetectionDataset::JSONDetectionDataset(ClassManager* class_manager) : Dataset(class_manager) {
	
}

JSONDetectionDataset::~JSONDetectionDataset() {
	
}

void JSONDetectionDataset::Load(JSON dataset_json, bool dont_load, DatasetLoadSelection selection) {
  // TODO Make those work
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
      } else {
        LOGDEBUG << "Registered class \"" << class_name << "\".";
      }
		}
    label_maps_ = 0;
    input_maps_ = 3;
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
      std::string boxes = element_json["bounding_boxes"];

      if(element_type.compare("tensor_stream") == 0) {
        filename = element_json["filename"];
        tensor_stream = TensorStream::FromFile(filename, class_manager_);
      } else if(element_type.compare("list") == 0) {
        filename = "LISTTENSORSTREAM";
        std::string imagelist_path = element_json["imagelist"];
        std::string labellist_path = "DONOTLOAD";
        std::string images = element_json["imagepath"];
        std::string labels = "DONOTLOAD";
        tensor_stream = new ListTensorStream(class_manager_);
        dynamic_cast<ListTensorStream*>(tensor_stream)->LoadFiles(imagelist_path, images, labellist_path, labels);
      }

			unsigned int tensor_count = tensor_stream->GetTensorCount();

      // Load bounding boxes
      std::ifstream boxes_file(boxes, std::ios::in);
      if(!boxes_file.good()) {
        FATAL("Cannot load bounding boxes from " << boxes);
      }

      JSON boxes_json = JSON::parse(boxes_file);
      if(boxes_json.count("samples") != 1 || !boxes_json["samples"].is_array()) {
        FATAL(boxes << " does not contain bounding boxes!");
      }

      JSON boxes_samples = boxes_json["samples"];
      if(tensor_count != boxes_samples.size() * 2) {
        FATAL(boxes << " contains the wrong number of samples. Needs " << tensor_count / 2 << ", contains " << boxes_samples.size() << ".");
      }

      // Validate tensor count
			if(tensor_count == 0) {
				LOGWARN << "Empty tensor stream for segment \"" << segment << "\", filename: " << filename;
			}

      if(tensor_count % 2 == 1) {
        FATAL("Wrong count (not divisible by 2) in tensor stream for segment \"" << segment << "\", filename: " << filename);
      }

			// Iterate through tensors to determine max dimensions
			for (unsigned int t = 0; t < tensor_count; t+=2) {
				if(tensor_stream->GetWidth(t) > max_width_)
					max_width_ = tensor_stream->GetWidth(t);
				
				if(tensor_stream->GetHeight(t) > max_height_)
					max_height_ = tensor_stream->GetHeight(t);

        TensorStreamAccessor accessor;
        accessor.tensor_stream = tensor_stream;
        accessor.sample_in_stream = t;

        // Load boxes
        JSON tensor_box = boxes_samples[t / 2];
        if(!tensor_box.is_object() || tensor_box.count("boxes") != 1 || !tensor_box["boxes"].is_array()) {
          FATAL("Wrong bounding box JSON for sample " << t / 2);
        }
        // LOGDEBUG << "Sample " << t / 2 << " has " << tensor_box["boxes"].size() << " bounding boxes.";

        for(unsigned int b = 0; b < tensor_box["boxes"].size(); b++) {
          JSON box_json = tensor_box["boxes"][b];
          BoundingBox box(box_json["x"], box_json["y"], box_json["w"], box_json["h"]);
          if(box_json.count("difficult") == 1 && box_json["difficult"].is_number()) {
            unsigned int difficult = box_json["difficult"];
            box.flag2 = difficult > 0;
          }

          // Find the class by name
          std::string class_name = box_json["class"];
          box.c = class_manager_->GetClassIdByName(class_name);
          bool class_found = box.c != UNKNOWN_CLASS;

          if(!class_found) {
            LOGDEBUG << "Autoregistering class " << class_name;
            class_manager_->RegisterClassByName(class_name, 0, 1.0);
            box.c = class_manager_->GetClassIdByName(class_name);
          }

          accessor.bounding_boxes_.push_back(std::move(box));
        }

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
			
      // Check if input map count needs to be set
      if(input_maps_ == 0) {
        if(tensor_stream->GetTensorCount() > 0) {
          input_maps_ = tensor_stream->GetMaps(0);
        }
      } else {
        if(tensor_stream->GetTensorCount() > 0) {
          if(input_maps_ != tensor_stream->GetMaps(0)) {
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

  // Go trough all accessors and normalize their bounding boxes
  for(unsigned int a = 0; a < training_accessors_.size(); a++) {
    const datum width = training_accessors_[a].tensor_stream->GetWidth(training_accessors_[a].sample_in_stream);
    const datum height = training_accessors_[a].tensor_stream->GetHeight(training_accessors_[a].sample_in_stream);
    for(unsigned int b = 0; b < training_accessors_[a].bounding_boxes_.size(); b++) {
      BoundingBox *box = &(training_accessors_[a].bounding_boxes_[b]);
      box->x /= width;
      box->y /= height;
      box->w /= width;
      box->h /= height;
    }
  }
  for(unsigned int a = 0; a < testing_accessors_.size(); a++) {
    const datum width = testing_accessors_[a].tensor_stream->GetWidth(testing_accessors_[a].sample_in_stream);
    const datum height = testing_accessors_[a].tensor_stream->GetHeight(testing_accessors_[a].sample_in_stream);
    for(unsigned int b = 0; b < testing_accessors_[a].bounding_boxes_.size(); b++) {
      BoundingBox *box = &(testing_accessors_[a].bounding_boxes_[b]);
      box->x /= width;
      box->y /= height;
      box->w /= width;
      box->h /= height;
    }
  }

  // Randomize order of accessors if specified
  int random_seed = 0;
  JSON_TRY_INT(random_seed, dataset_json, "training_random_seed", 0);
  if(random_seed != 0) {
    LOGINFO << "Randomizing training samples using seed " << random_seed;
    std::mt19937_64 training_randomizer(random_seed);
    std::shuffle(training_accessors_.begin(), training_accessors_.end(), training_randomizer);
  }
}	


bool JSONDetectionDataset::GetTrainingSample(Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
  UNREFERENCED_PARAMETER(label_tensor);
  UNREFERENCED_PARAMETER(helper_tensor);
  if (index < tensor_count_training_) {
    bool success = true;

    TensorStream* training_stream = training_accessors_[index].tensor_stream;
    unsigned int index_image = training_accessors_[index].sample_in_stream;

    success &= training_stream->CopySample(index_image, 0, data_tensor, sample, true);

    weight_tensor.Clear (1.0, sample);

    return success;
  } else return false;
}

bool JSONDetectionDataset::GetTestingSample(Tensor& data_tensor, Tensor& label_tensor,Tensor& helper_tensor, Tensor& weight_tensor,  unsigned int sample, unsigned int index) {
  UNREFERENCED_PARAMETER(label_tensor);
  UNREFERENCED_PARAMETER(helper_tensor);
  if (index < tensor_count_testing_) {
    bool success = true;

    TensorStream* testing_stream = testing_accessors_[index].tensor_stream;
    unsigned int index_image = testing_accessors_[index].sample_in_stream;

    success &= testing_stream->CopySample(index_image, 0, data_tensor, sample, true);

    weight_tensor.Clear (1.0, sample);

    return success;
  } else return false;
}

bool JSONDetectionDataset::GetTrainingMetadata(DatasetMetadataPointer *metadata_array, unsigned int sample,
                                               unsigned int index) {
  if (index < tensor_count_training_) {
    metadata_array[sample] = &(training_accessors_[index].bounding_boxes_);
    return true;
  } else return false;
}

bool JSONDetectionDataset::GetTestingMetadata(DatasetMetadataPointer *metadata_array, unsigned int sample,
                                              unsigned int index) {
  if (index < tensor_count_testing_) {
    metadata_array[sample] = &(testing_accessors_[index].bounding_boxes_);
    return true;
  } else return false;
}

}
