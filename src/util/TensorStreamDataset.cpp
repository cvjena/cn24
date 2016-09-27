/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifdef BUILD_POSIX
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#endif

#include <fstream>
#include <cstdlib>

#include <sstream>

#include "Config.h"
#include "Dataset.h"
#include "Init.h"

#include "KITTIData.h"
#include "ConfigParsing.h"

#include "FloatTensorStream.h"

namespace Conv {

TensorStreamDataset::TensorStreamDataset (
    /*
    std::istream& training_stream,
    std::istream& testing_stream,
    */
    TensorStream* training_stream,
    TensorStream* testing_stream,
    unsigned int classes,
    std::vector< std::string > class_names,
    std::vector<unsigned int> class_colors,
		std::vector<datum> class_weights,
    ClassManager* class_manager,
    dataset_localized_error_function error_function) : Dataset(class_manager),
    training_stream_(training_stream), testing_stream_(testing_stream),
  class_names_ (class_names), class_colors_ (class_colors),
	class_weights_(class_weights),classes_ (classes),
  error_function_ (error_function) {
  LOGDEBUG << "Instance created.";

  if (classes != class_names.size() ||
      classes != class_colors.size()) {
    FATAL ("Class count does not match class information count!");
  }

  
  // Count tensors
  Tensor tensor;

  tensor_count_training_ = training_stream_->GetTensorCount();
  LOGDEBUG << tensor_count_training_  / 2 << " training tensors";

  // We need alternating label and image tensors, so we need an even count
  if (tensor_count_training_ & 1) {
    FATAL ("Odd training tensor count!");
  }

  tensor_count_testing_ = testing_stream->GetTensorCount();
  LOGDEBUG << tensor_count_testing_ / 2 << " testing tensors";

  if (tensor_count_testing_ & 1) {
    FATAL ("Odd testing tensor count!");
  }

  tensors_ = (tensor_count_testing_ + tensor_count_training_) / 2;

  // Read tensors
  max_width_ = 0;
  max_height_ = 0;
  
  if((tensor_count_training_ + tensor_count_testing_) > 0) {
    LOGINFO << "Loaded " << (tensor_count_training_ + tensor_count_testing_) / 2 << " Tensors.";
  }

  for (unsigned int t = 0; t < (tensor_count_training_ / 2); t++) {
    if(training_stream_->GetWidth(2*t) > max_width_)
      max_width_ = training_stream_->GetWidth(2*t);
    
    if(training_stream_->GetHeight(2*t) > max_height_)
      max_height_ = training_stream_->GetHeight(2*t);
  }

  for (unsigned int t = (tensor_count_training_ / 2) ; t < tensors_; t++) {
    if(testing_stream_->GetWidth(2*t) > max_width_)
      max_width_ = testing_stream_->GetWidth(2*t);
    
    if(testing_stream_->GetHeight(2*t) > max_height_)
      max_height_ = testing_stream_->GetHeight(2*t);
  }

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
  
  if(training_stream_->GetTensorCount() > 0) {
    input_maps_ = training_stream_->GetMaps(0);
    label_maps_ = training_stream_->GetMaps(1);
  } else {
    input_maps_ = testing_stream_->GetMaps(0);
    label_maps_ = testing_stream_->GetMaps(1);
  }

  // Prepare error cache
  error_cache.Resize (1, max_width_, max_height_, 1);

  for (unsigned int y = 0; y < max_height_; y++) {
    for (unsigned int x = 0; x < max_width_; x++) {
      *error_cache.data_ptr (x, y) = error_function (x, y, max_width_, max_height_);
    }
  }

  // System::viewer->show(&error_cache);
}

Task TensorStreamDataset::GetTask() const {
  return Task::SEMANTIC_SEGMENTATION;
}

unsigned int TensorStreamDataset::GetWidth() const {
  return max_width_;
}

unsigned int TensorStreamDataset::GetHeight() const {
  return max_height_;
}

unsigned int TensorStreamDataset::GetInputMaps() const {
  return input_maps_;
}

unsigned int TensorStreamDataset::GetLabelMaps() const {
  return label_maps_;
}

unsigned int TensorStreamDataset::GetTrainingSamples() const {
  return tensor_count_training_ / 2;
}

unsigned int TensorStreamDataset::GetTestingSamples() const {
  return tensor_count_testing_ / 2;
}

bool TensorStreamDataset::SupportsTesting() const {
  return tensor_count_testing_ > 0;
}

bool TensorStreamDataset::GetTrainingSample (Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
  if (index < tensor_count_training_ / 2) {
    bool success = true;
    success &= training_stream_->CopySample(2 * index, 0, data_tensor, sample);
    success &= training_stream_->CopySample(2 * index + 1, 0, label_tensor, sample);

    unsigned int data_width = training_stream_->GetWidth(2 * index);
    unsigned int data_height = training_stream_->GetHeight(2 * index);
    
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

    //if (data_width == GetWidth() && data_height == GetHeight()) {
    //  success &= Tensor::CopySample (error_cache, 0, weight_tensor, sample);
    //} else {
      // Reevaluate error function
      weight_tensor.Clear (0.0, sample);

      #pragma omp parallel for default(shared)
      for (unsigned int y = 0; y < data_height; y++) {
        for (unsigned int x = 0; x < data_width; x++) {
					const datum class_weight = class_weights_[label_tensor.PixelMaximum(x, y, sample)];
          *weight_tensor.data_ptr (x, y, 0, sample) = error_function_ (x, y, data_width, data_height) * class_weight;
        }
      }
    //}

    return success;
  } else return false;
}

bool TensorStreamDataset::GetTestingSample (Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
  if (index < tensor_count_testing_ / 2) {
    bool success = true;
    success &= testing_stream_->CopySample(2 * index, 0, data_tensor, sample);
    success &= testing_stream_->CopySample(2 * index + 1, 0, label_tensor, sample);
    
    unsigned int data_width = testing_stream_->GetWidth(2 * index);
    unsigned int data_height = testing_stream_->GetHeight(2 * index);

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

    //if (data_width == GetWidth() && data_height == GetHeight()) {
    //  success &= Tensor::CopySample (error_cache, 0, weight_tensor, sample);
    //} else {
      // Reevaluate error function
      weight_tensor.Clear (0.0, sample);

      #pragma omp parallel for default(shared)
      for (unsigned int y = 0; y < data_height; y++) {
        for (unsigned int x = 0; x < data_width; x++) {
					const datum class_weight = class_weights_[label_tensor.PixelMaximum(x, y, sample)];
          *weight_tensor.data_ptr (x, y, 0, sample) = error_function_ (x, y, data_width, data_height) * class_weight;
        }
      }
    //}

    return success;
  } else return false;
}

TensorStreamDataset* TensorStreamDataset::CreateFromConfiguration (std::istream& file , bool dont_load, DatasetLoadSelection selection, ClassManager* class_manager) {
  unsigned int classes = 0;
  std::vector<std::string> class_names;
  std::vector<unsigned int> class_colors;
	std::vector<datum> class_weights;
  dataset_localized_error_function error_function = DefaultLocalizedErrorFunction;
  std::string training_file;
  std::string testing_file;
  
  TensorStream* training_stream = new FloatTensorStream();
  TensorStream* testing_stream = new FloatTensorStream();

  file.clear();
  file.seekg (0, std::ios::beg);

  while (! file.eof()) {
    std::string line;
    std::getline (file, line);
    
    if (StartsWithIdentifier (line, "classes")) {
      ParseCountIfPossible (line, "classes", classes);

      if (classes != 0) {
        for (unsigned int c = 0; c < classes; c++) {
          std::string class_name;
          std::getline (file, class_name);
          class_names.push_back (class_name);
        }
      }
    }

    if (StartsWithIdentifier (line, "colors")) {
      if (classes != 0) {
        for (unsigned int c = 0; c < classes; c++) {
          std::string color;
          std::getline (file, color);
          unsigned long color_val_l = std::strtoul (color.c_str(), nullptr, 16);

          if (color_val_l < 0x100000000L) {
            class_colors.push_back ( (unsigned int) color_val_l);
          } else {
            FATAL ("Not a valid color!");
          }
        }
      }
    }

		if (StartsWithIdentifier(line, "weights")) {
			if (classes != 0) {
				for (unsigned int c = 0; c < classes; c++) {
					std::string weight;
					datum dweight;
					std::getline(file, weight);
					std::stringstream ss;
					ss << weight;
					ss >> dweight;
					class_weights.push_back(dweight);
					LOGDEBUG << "Class " << c << " weight: " << dweight;
				}
			}
		}

    if (StartsWithIdentifier (line, "localized_error")) {
      std::string error_function_name;
      ParseStringIfPossible (line, "localized_error", error_function_name);

      if (error_function_name.compare ("kitti") == 0) {
        LOGDEBUG << "Loading dataset with KITTI error function";
        error_function = KITTIData::LocalizedError;
      } else if (error_function_name.compare ("default")) {
        LOGDEBUG << "Loading dataset with KITTI error function";
        error_function = DefaultLocalizedErrorFunction;
      }
    }

    ParseStringIfPossible (line, "training", training_file);
    ParseStringIfPossible (line, "testing", testing_file);
  }

  LOGDEBUG << "Loading dataset with " << classes << " classes";
  LOGDEBUG << "Training tensor: " << training_file;
  LOGDEBUG << "Testing tensor: " << testing_file;
	
	if (class_weights.size() != classes) {
		for (unsigned int c = 0; c < classes; c++)
			class_weights.push_back(1.0);
	}
	
  if (!dont_load && (selection == LOAD_BOTH || selection == LOAD_TRAINING_ONLY) && training_file.length() > 0) {
    training_stream = TensorStream::FromFile(training_file, class_manager);
  } else {
  }

  if (!dont_load && (selection == LOAD_BOTH || selection == LOAD_TESTING_ONLY) && testing_file.length() > 0) {
    testing_stream = TensorStream::FromFile(testing_file, class_manager);
  } else {
  }

  return new TensorStreamDataset (training_stream, testing_stream, classes,
                                  class_names, class_colors, class_weights, class_manager, error_function);
}

}
