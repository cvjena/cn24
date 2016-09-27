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
#include <cstring>
#include <sstream>

#include "Config.h"
#include "Dataset.h"
#include "Init.h"

#include "KITTIData.h"
#include "TensorViewer.h"
#include "ConfigParsing.h"

namespace Conv {
datum DefaultLocalizedErrorFunction (unsigned int x, unsigned int y, unsigned int w, unsigned int h) {
  UNREFERENCED_PARAMETER(x);
  UNREFERENCED_PARAMETER(y);
  UNREFERENCED_PARAMETER(w);
  UNREFERENCED_PARAMETER(h);
  return 1;
}
TensorStreamPatchDataset::TensorStreamPatchDataset(std::istream& training_stream,
		std::istream& testing_stream,
		unsigned int classes,
		std::vector< std::string > class_names,
		std::vector<unsigned int> class_colors,
		std::vector<datum> class_weights,
		unsigned int patchsize_x,
		unsigned int patchsize_y, ClassManager* class_manager,
		dataset_localized_error_function error_function,
    int training_fd, int testing_fd ) : Dataset(class_manager),
  patchsize_x_(patchsize_x), patchsize_y_(patchsize_y),
	class_names_(class_names), class_colors_(class_colors),
	class_weights_(class_weights),
  classes_(classes),
	error_function_(error_function) {
	LOGDEBUG << "Instance created.";

	if (classes != class_names.size() ||
		classes != class_colors.size()) {
		FATAL("Class count does not match class information count!");
	}

	// Count tensors
	Tensor tensor;

	while (!training_stream.eof()) {
		tensor.Deserialize(training_stream, true);

		if (tensor.elements() == 0)
			break;

		// LOGDEBUG << "Tensor " << tensor_count_training_ << ": " << tensor;
		tensor_count_training_++;

		training_stream.peek();
	}

	LOGDEBUG << tensor_count_training_ / 2 << " training tensors";

	// We need alternating label and image tensors, so we need an even count
	if (tensor_count_training_ & 1) {
		FATAL("Odd training tensor count!");
	}

	while (!testing_stream.eof()) {
		tensor.Deserialize(testing_stream, true);

		if (tensor.elements() == 0)
			break;

		// LOGDEBUG << "Tensor " << tensor_count_testing_ << ": " << tensor;
		tensor_count_testing_++;

		testing_stream.peek();
	}

	LOGDEBUG << tensor_count_testing_ / 2 << " testing tensors";

	if (tensor_count_testing_ & 1) {
		FATAL("Odd testing tensor count!");
	}

	tensors_ = (tensor_count_testing_ + tensor_count_training_) / 2;

	// Reset streams
	training_stream.clear();
	testing_stream.clear();
	training_stream.seekg(0, std::ios::beg);
	testing_stream.seekg(0, std::ios::beg);

	// Allocate arrays that depend on the tensor count
	if (tensors_ > 0) {
		data_ = new Tensor[tensors_];
		labels_ = new Tensor[tensors_];
		last_sample_ = new unsigned int[tensors_];
	}
	else {
		data_ = new Tensor[1];
		labels_ = new Tensor[1];
		last_sample_ = new unsigned int[1];
	}

	// Read tensors
  if((tensor_count_training_ + tensor_count_testing_) > 0) {
    LOGINFO << "Deserializing " << (tensor_count_training_ + tensor_count_testing_) / 2 << " Tensors..." << std::endl << std::flush;
  }

	for (unsigned int t = 0; t < (tensor_count_training_ / 2); t++) {
		data_[t].Deserialize(training_stream, false, true, training_fd);

		unsigned int inner_width = data_[t].width() - (patchsize_x_ - 1);
		unsigned int inner_height = data_[t].height() - (patchsize_y_ - 1);

		if (t == 0)
			last_sample_[t] = inner_width * inner_height;
		else
			last_sample_[t] = last_sample_[t - 1] + (inner_width * inner_height);

		sample_count_training_ += inner_width * inner_height;

		labels_[t].Deserialize(training_stream, false, true, training_fd);
    
    std::cout << "." << std::flush;
	}

	for (unsigned int t = (tensor_count_training_ / 2); t < tensors_; t++) {
		data_[t].Deserialize(testing_stream, false, true, testing_fd);

		unsigned int inner_width = data_[t].width() - (patchsize_x_ - 1);
		unsigned int inner_height = data_[t].height() - (patchsize_y_ - 1);

		if (t == 0)
			last_sample_[t] = inner_width * inner_height;
		else
			last_sample_[t] = last_sample_[t - 1] + (inner_width * inner_height);

		sample_count_testing_ += inner_width * inner_height;

		labels_[t].Deserialize(testing_stream, false, true, testing_fd);
    
    std::cout << "." << std::flush;
	}

	input_maps_ = data_[0].maps();
	label_maps_ = labels_[0].maps();
}

Task TensorStreamPatchDataset::GetTask() const {
  return Task::SEMANTIC_SEGMENTATION;
}

unsigned int TensorStreamPatchDataset::GetWidth() const {
  return patchsize_x_;
}

unsigned int TensorStreamPatchDataset::GetHeight() const {
  return patchsize_y_;
}

unsigned int TensorStreamPatchDataset::GetInputMaps() const {
  return input_maps_;
}

unsigned int TensorStreamPatchDataset::GetLabelMaps() const {
  return label_maps_;
}

unsigned int TensorStreamPatchDataset::GetTrainingSamples() const {
  return sample_count_training_;
}

unsigned int TensorStreamPatchDataset::GetTestingSamples() const {
  return sample_count_testing_;
}

bool TensorStreamPatchDataset::SupportsTesting() const {
  return tensor_count_testing_ > 0;
}

bool TensorStreamPatchDataset::GetTrainingSample (Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
  if (index < sample_count_training_) {
    bool success = true;

    // Find patch
    unsigned int t = 0;

    while ( (t < tensors_) && (index >= last_sample_[t]))
      t++;

    if (index >= last_sample_[t])
      return false;

    unsigned int inner_width = data_[t].width() - (patchsize_x_ - 1);
    unsigned int inner_height = data_[t].height() - (patchsize_y_ - 1);
    
    unsigned int first_sample = last_sample_[t] - inner_width * inner_height;
    unsigned int sample_offset = index - first_sample;

    // Find x and y coords
    unsigned int row = sample_offset / (data_[t].width() - (patchsize_x_ - 1));
    unsigned int col = sample_offset - (row * (data_[t].width() - (patchsize_x_ - 1)));

    // Copy patch
    for (unsigned int map = 0; map < input_maps_; map++) {
      for (unsigned int y = 0; y < patchsize_y_; y++) {
        const datum* row_ptr = data_[t].data_ptr_const (col, row + y, map, 0);
        datum* target_row_ptr = data_tensor.data_ptr (0, y, map, sample);
        std::memcpy (target_row_ptr, row_ptr, patchsize_x_ * sizeof(datum) / sizeof (char));
      }
    }
    
    // Copy label
    for (unsigned int map = 0; map < label_maps_; map++) {
      *label_tensor.data_ptr (0, 0, map, sample) =
        *labels_[t].data_ptr_const (col + (patchsize_x_ / 2), row + (patchsize_y_ / 2), map, 0);
    }

		// Copy helper tensor
		if (data_[t].width() > 1)
			*helper_tensor.data_ptr(0, 0, 0, sample) = ((datum)col) / ((datum)data_[t].width() - 1);
		else
			*helper_tensor.data_ptr(0, 0, 0, sample) = 0;

		if (data_[t].height() > 1)
			*helper_tensor.data_ptr(0, 0, 1, sample) = ((datum)row) / ((datum)data_[t].height() - 1);
		else
			*helper_tensor.data_ptr(0, 0, 1, sample) = 0;

		const datum class_weight = class_weights_[label_tensor.PixelMaximum(0, 0, sample)];

    // Copy error
    *weight_tensor.data_ptr (0, 0, 0, sample) =
      error_function_ (col + (patchsize_x_ / 2), row + (patchsize_y_ / 2),
      data_[t].width(), data_[t].height()) * class_weight;

    return success;
  } else return false;
}

bool TensorStreamPatchDataset::GetTestingSample (Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
  if (index < sample_count_testing_) {
		index += sample_count_testing_;
    bool success = true;

    // Find patch
    unsigned int t = 0;

    while ( (t < tensors_) && (index >= last_sample_[t]))
      t++;

    if (index >= last_sample_[t])
      return false;

    unsigned int inner_width = data_[t].width() - (patchsize_x_ - 1);
    unsigned int inner_height = data_[t].height() - (patchsize_y_ - 1);
    
    unsigned int first_sample = last_sample_[t] - inner_width * inner_height;
    unsigned int sample_offset = index - first_sample;

    // Find x and y coords
    unsigned int row = sample_offset / (data_[t].width() - (patchsize_x_ - 1));
    unsigned int col = sample_offset - (row * (data_[t].width() - (patchsize_x_ - 1)));

    // Copy patch
    for (unsigned int map = 0; map < input_maps_; map++) {
      for (unsigned int y = 0; y < patchsize_y_; y++) {
        const datum* row_ptr = data_[t].data_ptr_const (col, row + y, map, 0);
        datum* target_row_ptr = data_tensor.data_ptr (0, y, map, sample);
        std::memcpy (target_row_ptr, row_ptr, patchsize_x_ * sizeof(datum) / sizeof (char));
      }
    }
    
    // Copy label
    for (unsigned int map = 0; map < label_maps_; map++) {
      *label_tensor.data_ptr (0, 0, map, sample) =
        *labels_[t].data_ptr_const (col + (patchsize_x_ / 2), row + (patchsize_y_ / 2), map, 0);
    }

		// Copy helper tensor
		if (data_[t].width() > 1)
			*helper_tensor.data_ptr(0, 0, 0, sample) = ((datum)col) / ((datum)data_[t].width() - 1);
		else
			*helper_tensor.data_ptr(0, 0, 0, sample) = 0;

		if (data_[t].height() > 1)
			*helper_tensor.data_ptr(0, 0, 1, sample) = ((datum)row) / ((datum)data_[t].height() - 1);
		else
			*helper_tensor.data_ptr(0, 0, 1, sample) = 0;

		const datum class_weight = class_weights_[label_tensor.PixelMaximum(0, 0, sample)];

    // Copy error
    *weight_tensor.data_ptr (0, 0, 0, sample) =
      error_function_ (col + (patchsize_x_ / 2), row + (patchsize_y_ / 2),
      data_[t].width(), data_[t].height()) * class_weight;

    return success;
  } else return false;
}

TensorStreamPatchDataset* TensorStreamPatchDataset::CreateFromConfiguration (std::istream& file , bool dont_load, DatasetLoadSelection selection, unsigned int patchsize_x, unsigned int patchsize_y, ClassManager* class_manager) {
  unsigned int classes = 0;
  std::vector<std::string> class_names;
  std::vector<unsigned int> class_colors;
	std::vector<datum> class_weights;
  dataset_localized_error_function error_function = DefaultLocalizedErrorFunction;
  std::string training_file;
  std::string testing_file;
  int training_fd = 0;
  int testing_fd = 0;
  bool no_mmap = false;

  file.clear();
  file.seekg (0, std::ios::beg);

  while (! file.eof()) {
    std::string line;
    std::getline (file, line);
    
    if (StartsWithIdentifier (line, "nommap")) {
      LOGDEBUG << "Dataset requested to not be memory mapped.";
      no_mmap = true;
    }

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

  std::istream* training_stream = nullptr;
  std::istream* testing_stream = nullptr;

  if (!dont_load && (selection == LOAD_BOTH || selection == LOAD_TRAINING_ONLY) && training_file.length() > 0) {
    training_stream = new std::ifstream (training_file, std::ios::in | std::ios::binary);
    if(!training_stream->good()) {
      FATAL("Failed to load " << training_file << "!");
    }
#ifdef BUILD_POSIX
    if(!no_mmap)
      training_fd = open(training_file.c_str(), O_RDONLY);
    if(training_fd < 0) {
      FATAL("Failed to load " << training_file << "!");
    }
#endif
  } else {
    training_stream = new std::istringstream();
  }

  if (!dont_load && (selection == LOAD_BOTH || selection == LOAD_TESTING_ONLY) && testing_file.length() > 0) {
    testing_stream = new std::ifstream (testing_file, std::ios::in | std::ios::binary);
    if(!testing_stream->good()) {
      FATAL("Failed to load " << testing_file << "!");
    }
#ifdef BUILD_POSIX
    if(!no_mmap)
      testing_fd = open(training_file.c_str(), O_RDONLY);
    if(testing_fd < 0) {
      FATAL("Failed to load " << testing_file << "!");
    }
#endif
  } else {
    testing_stream = new std::istringstream();
  }

	if (class_weights.size() != classes) {
		for (unsigned int c = 0; c < classes; c++)
			class_weights.push_back(1.0);
	}

  return new TensorStreamPatchDataset (*training_stream, *testing_stream, classes,
                                       class_names, class_colors, class_weights, patchsize_x,
                                       patchsize_y, class_manager, error_function, training_fd, testing_fd);
}

}
