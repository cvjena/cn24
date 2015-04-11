/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

/**
* @file DatasetInputLayer.cpp
* @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
*/


#include <vector>
#include <array>
#include <random>
#include <algorithm>
#include <cstring>

#include "NetGraph.h"
#include "DatasetInputLayer.h"

namespace Conv {

DatasetInputLayer::DatasetInputLayer (Dataset& dataset,
                                      const unsigned int batch_size,
                                      const datum loss_sampling_p,
                                      const unsigned int seed) :
  dataset_ (dataset), batch_size_ (batch_size),
  loss_sampling_p_ (loss_sampling_p),
  seed_ (seed), generator_ (seed), dist_ (0.0, 1.0) {
  LOGDEBUG << "Instance created.";

  label_maps_ = dataset_.GetLabelMaps();
  input_maps_ = dataset_.GetInputMaps();

  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }

  if(dataset_.GetMethod() == FCN)
    LOGDEBUG << "Using loss sampling probability: " << loss_sampling_p_;

  elements_training_ = dataset_.GetTrainingSamples();
  elements_testing_ = dataset_.GetTestingSamples();
  elements_total_ = elements_training_ + elements_testing_;
  
  LOGDEBUG << "Total samples: " << elements_total_;

  // Generate random permutation of the samples
  // First, we need an array of ascending numbers
  for (unsigned int i = 0; i < elements_training_; i++) {
    perm_.push_back (i);
  }

  RedoPermutation();
}

bool DatasetInputLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                                       std::vector< CombinedTensor* >& outputs) {
  if (inputs.size() != 0) {
    LOGERROR << "Inputs specified but not supported";
    return false;
  }

  if (dataset_.GetMethod() == FCN) {
    CombinedTensor* data_output =
      new CombinedTensor (batch_size_, dataset_.GetWidth(),
                          dataset_.GetHeight(), input_maps_);

    CombinedTensor* label_output =
      new CombinedTensor (batch_size_, dataset_.GetWidth(),
                          dataset_.GetHeight(), label_maps_);

    CombinedTensor* helper_output =
      new CombinedTensor (batch_size_, dataset_.GetWidth(),
                          dataset_.GetHeight(), 2);

    CombinedTensor* localized_error_output =
      new CombinedTensor (batch_size_, dataset_.GetWidth(),
                          dataset_.GetHeight(), 1);

    outputs.push_back (data_output);
    outputs.push_back (label_output);
    outputs.push_back (helper_output);
    outputs.push_back (localized_error_output);
  } else if (dataset_.GetMethod() == PATCH) {
    CombinedTensor* data_output =
      new CombinedTensor (batch_size_, dataset_.GetWidth(),
                          dataset_.GetHeight(), input_maps_);

    CombinedTensor* label_output =
      new CombinedTensor (batch_size_, 1,
                          1, label_maps_);

    CombinedTensor* helper_output =
      new CombinedTensor (batch_size_, 1,
                          1, 2);

    CombinedTensor* localized_error_output =
      new CombinedTensor (batch_size_, 1,
                          1, 1);

    outputs.push_back (data_output);
    outputs.push_back (label_output);
    outputs.push_back (helper_output);
    outputs.push_back (localized_error_output);
  }

  return true;
}

bool DatasetInputLayer::Connect (const std::vector< CombinedTensor* >& inputs,
                                 const std::vector< CombinedTensor* >& outputs,
                                 const NetStatus* net) {
  // TODO validate
  CombinedTensor* data_output = outputs[0];
  CombinedTensor* label_output = outputs[1];
  CombinedTensor* helper_output = outputs[2];
  CombinedTensor* localized_error_output = outputs[3];

  if (data_output == nullptr || label_output == nullptr ||
      localized_error_output == nullptr)
    return false;

  bool valid = inputs.size() == 0 && outputs.size() == 4;

  if (valid) {
    data_output_ = data_output;
    label_output_ = label_output;
    helper_output_ = helper_output;
    localized_error_output_ = localized_error_output;
  }

  return valid;
}

void DatasetInputLayer::FeedForward() {
#ifdef BUILD_OPENCL
  data_output_->data.MoveToCPU (true);
  label_output_->data.MoveToCPU (true);
  localized_error_output_->data.MoveToCPU (true);
#endif

  for (std::size_t sample = 0; sample < batch_size_; sample++) {
    unsigned int selected_element = 0;
    bool force_no_weight = false;

    if (testing_) {
      // The testing samples are not randomized
      selected_element = current_element_testing_;
      current_element_testing_++;

      if (current_element_testing_ >= elements_testing_) {
        force_no_weight = true;
        selected_element = 0;
      }
    } else {
      // Select samples until one from the right subset is hit
      // Select a sample from the permutation
      selected_element = perm_[current_element_];

      // Select next element
      current_element_++;

      // If this is is out of bounds, start at the beginning and randomize
      // again.
      if (current_element_ >= perm_.size()) {
        current_element_ = 0;
        RedoPermutation();
      }
    }

    // Copy image and label
    bool success;

    if (testing_)
      success = dataset_.GetTestingSample (data_output_->data, label_output_->data, localized_error_output_->data, sample, selected_element);
    else
      success = dataset_.GetTrainingSample (data_output_->data, label_output_->data, localized_error_output_->data, sample, selected_element);

    if (!success) {
      FATAL ("Cannot load samples from Dataset!");
    }

    if (!testing_ && !force_no_weight && dataset_.GetMethod() == FCN) {
      // Perform loss sampling
#ifdef BUILD_OPENCL
      localized_error_output_->data.MoveToGPU();
#endif
      const unsigned int block_size = 12;

      for (unsigned int y = 0; y < localized_error_output_->data.height(); y += block_size) {
        for (unsigned int x = 0; x < localized_error_output_->data.width(); x += block_size) {
          if (dist_ (generator_) > loss_sampling_p_) {
            for (unsigned int iy = y; iy < y + block_size && iy < localized_error_output_->data.height(); iy++) {
              for (unsigned int ix = x; ix < x + block_size && ix < localized_error_output_->data.width(); ix++) {
                *localized_error_output_->data.data_ptr (ix, iy, 0, sample) = 0;
              }
            }
          }
        }
      }
    }

    // Copy localized error
    if (force_no_weight)
      localized_error_output_->data.Clear (0.0, sample);
  }
}

void DatasetInputLayer::BackPropagate() {
  // No inputs, no backprop.
}

unsigned int DatasetInputLayer::GetBatchSize() {
  return batch_size_;
}

unsigned int DatasetInputLayer::GetLabelWidth() {
  return (dataset_.GetMethod() == PATCH) ? 1 : dataset_.GetWidth();
}

unsigned int DatasetInputLayer::GetLabelHeight() {
  return (dataset_.GetMethod() == PATCH) ? 1 : dataset_.GetHeight();
}

unsigned int DatasetInputLayer::GetSamplesInTestingSet() {
  return dataset_.GetTestingSamples();
}

unsigned int DatasetInputLayer::GetSamplesInTrainingSet() {
  return dataset_.GetTrainingSamples();
}

void DatasetInputLayer::RedoPermutation() {
  // Shuffle the array
  std::shuffle (perm_.begin(), perm_.end(), generator_);
}

void DatasetInputLayer::SetTestingMode (bool testing) {
  if (testing != testing_) {
    if (testing) {
      LOGDEBUG << "Enabled testing mode.";

      // Always test the same elements for consistency
      current_element_testing_ = 0;
    } else {
      LOGDEBUG << "Enabled training mode.";
    }
  }

  testing_ = testing;
}

void DatasetInputLayer::CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers) {
	NetGraphBuffer data_buffer;
	NetGraphBuffer label_buffer;
	NetGraphBuffer helper_buffer;
	NetGraphBuffer weight_buffer;
	data_buffer.description = "Data Output";
	label_buffer.description = "Label";
	helper_buffer.description = "Helper";
	weight_buffer.description = "Weight";
	buffers.push_back(data_buffer);
	buffers.push_back(label_buffer);
	buffers.push_back(helper_buffer);
	buffers.push_back(weight_buffer);
}

bool DatasetInputLayer::IsOpenCLAware() {
#ifdef BUILD_OPENCL
  return true;
#else
  return false;
#endif
}


}
