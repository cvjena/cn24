/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

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
  return 1;
}
TensorStreamPatchDataset::TensorStreamPatchDataset (std::istream& training_stream,
    std::istream& testing_stream,
    unsigned int classes,
    std::vector< std::string > class_names,
    std::vector<unsigned int> class_colors,
    unsigned int patchsize_x,
    unsigned int patchsize_y,
    dataset_localized_error_function error_function) :
  classes_ (classes), class_names_ (class_names), class_colors_ (class_colors),
  patchsize_x_ (patchsize_x), patchsize_y_ (patchsize_y),
  error_function_ (error_function) {
  LOGDEBUG << "Instance created.";

  if (classes != class_names.size() ||
      classes != class_colors.size()) {
    FATAL ("Class count does not match class information count!");
  }

  // Count tensors
  Tensor tensor;

  while (!training_stream.eof()) {
    tensor.Deserialize (training_stream);

    if (tensor.elements() == 0)
      break;

    // LOGDEBUG << "Tensor " << tensor_count_training_ << ": " << tensor;
    tensor_count_training_++;

    training_stream.peek();
  }

  LOGDEBUG << tensor_count_training_  / 2 << " training tensors";

  // We need alternating label and image tensors, so we need an even count
  if (tensor_count_training_ & 1) {
    FATAL ("Odd training tensor count!");
  }

  while (!testing_stream.eof()) {
    tensor.Deserialize (testing_stream);

    if (tensor.elements() == 0)
      break;

    // LOGDEBUG << "Tensor " << tensor_count_testing_ << ": " << tensor;
    tensor_count_testing_++;

    testing_stream.peek();
  }

  LOGDEBUG << tensor_count_testing_ / 2 << " testing tensors";

  if (tensor_count_testing_ & 1) {
    FATAL ("Odd testing tensor count!");
  }

  tensors_ = (tensor_count_testing_ + tensor_count_training_) / 2;

  // Reset streams
  training_stream.clear();
  testing_stream.clear();
  training_stream.seekg (0, std::ios::beg);
  testing_stream.seekg (0, std::ios::beg);

  // Allocate arrays that depend on the tensor count
  if (tensors_ > 0) {
    data_ = new Tensor[tensors_];
    labels_ = new Tensor[tensors_];
    last_sample_ = new unsigned int [tensors_];
  } else {
    data_ = new Tensor[1];
    labels_ = new Tensor[1];
    last_sample_ = new unsigned int [1];
  }

  // Read tensors
  unsigned int e = 0;

  for (unsigned int t = 0; t < (tensor_count_training_ / 2); t++) {
    data_[t].Deserialize (training_stream);

    unsigned int inner_width = data_[t].width() - (patchsize_x_ - 1);
    unsigned int inner_height = data_[t].height() - (patchsize_y_ - 1);

    if (t == 0)
      last_sample_[t] = inner_width * inner_height;
    else
      last_sample_[t] = last_sample_[t - 1] + (inner_width * inner_height);

    sample_count_training_ += inner_width * inner_height;

    labels_[t].Deserialize (training_stream);
  }

  for (unsigned int t = (tensor_count_training_ / 2) ; t < tensors_; t++) {
    data_[t].Deserialize (testing_stream);

    unsigned int inner_width = data_[t].width() - (patchsize_x_ - 1);
    unsigned int inner_height = data_[t].height() - (patchsize_y_ - 1);

    if (t == 0)
      last_sample_[t] = inner_width * inner_height;
    else
      last_sample_[t] = last_sample_[t - 1] + (inner_width * inner_height);

    sample_count_testing_ += inner_width * inner_height;

    labels_[t].Deserialize (testing_stream);
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

unsigned int TensorStreamPatchDataset::GetClasses() const {
  return classes_;
}

std::vector<std::string> TensorStreamPatchDataset::GetClassNames() const {
  return class_names_;
}

std::vector<unsigned int> TensorStreamPatchDataset::GetClassColors() const {
  return class_colors_;
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

    // Copy error
    *weight_tensor.data_ptr (0, 0, 0, sample) =
      error_function_ (col + (patchsize_x_ / 2), row + (patchsize_y_ / 2),
      data_[t].width(), data_[t].height());

    return success;
  } else return false;
}

bool TensorStreamPatchDataset::GetTestingSample (Tensor& data_tensor, Tensor& label_tensor, Tensor& helper_tensor, Tensor& weight_tensor, unsigned int sample, unsigned int index) {
  LOGERROR << "Patch testing is NIY, please use FCN for testing.";
  return false;
}

TensorStreamPatchDataset* TensorStreamPatchDataset::CreateFromConfiguration (std::istream& file , bool dont_load, DatasetLoadSelection selection, unsigned int patchsize_x, unsigned int patchsize_y) {
  unsigned int classes = 0;
  std::vector<std::string> class_names;
  std::vector<unsigned int> class_colors;
  dataset_localized_error_function error_function = DefaultLocalizedErrorFunction;
  std::string training_file;
  std::string testing_file;

  file.clear();
  file.seekg (0, std::ios::beg);

  while (! file.eof()) {
    std::string line;
    std::getline (file, line);

    if (StartsWithIdentifier (line, "classes")) {
      ParseCountIfPossible (line, "classes", classes);

      if (classes != 0) {
        for (int c = 0; c < classes; c++) {
          std::string class_name;
          std::getline (file, class_name);
          class_names.push_back (class_name);
        }
      }
    }

    if (StartsWithIdentifier (line, "colors")) {
      if (classes != 0) {
        for (int c = 0; c < classes; c++) {
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

  if (!dont_load && (selection == LOAD_BOTH || selection == LOAD_TRAINING_ONLY)) {
    training_stream = new std::ifstream (training_file, std::ios::in | std::ios::binary);
  } else {
    training_stream = new std::istringstream();
  }

  if (!dont_load && (selection == LOAD_BOTH || selection == LOAD_TESTING_ONLY)) {
    testing_stream = new std::ifstream (testing_file, std::ios::in | std::ios::binary);
  } else {
    testing_stream = new std::istringstream();
  }

  return new TensorStreamPatchDataset (*training_stream, *testing_stream, classes,
                                       class_names, class_colors, patchsize_x,
                                       patchsize_y, error_function);
}

}
