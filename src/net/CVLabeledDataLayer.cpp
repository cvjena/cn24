/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <vector>
#include <array>
#include <random>
#include <algorithm>

#include "Log.h"

#include "Tensor.h"
#include "CombinedTensor.h"

#include "CVLabeledDataLayer.h"

namespace Conv {

CVLabeledDataLayer::CVLabeledDataLayer (Tensor& data, Tensor& labels,
                                    const unsigned int batch_size,
                                    const int seed,
                                    const unsigned int split
                                   ) :
  data_ (std::move (data)), labels_ (std::move (labels)),
  batch_size_ (batch_size), seed_ (seed), generator_ (seed) {
  // Check if sample count matches
  if (data_.samples() != labels_.samples()) {
    FATAL ("The number of samples don't match. data: " << data_ <<
           ", labels: " << labels_);
  }

  LOGDEBUG << "Instance created: " << data_ << ", " << labels_;

  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }

  // Generate random permutation of the samples
  // First, we need an array of ascending numbers
  for (unsigned int i = 0; i < data_.samples(); i++) {
    perm_.push_back (i);
  }


  RedoPermutation();

  SetCrossValidationSplit (split);
  SetCrossValidationTestingSubset (0);
}


bool CVLabeledDataLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs) {
  if (inputs.size() != 0) {
    LOGERROR << "Inputs specified but not supported";
    return false;
  }

  CombinedTensor* data_output =
    new CombinedTensor (batch_size_, data_.width(),
                        data_.height(), data_.maps());

  CombinedTensor* label_output =
    new CombinedTensor (batch_size_, labels_.width(),
                        labels_.height(), labels_.maps());

  outputs.push_back (data_output);
  outputs.push_back (label_output);
  return true;
}

bool CVLabeledDataLayer::Connect (
  const std::vector< CombinedTensor* >& inputs,
  const std::vector< CombinedTensor* >& outputs) {
  CombinedTensor* data_output = outputs[0];
  CombinedTensor* label_output = outputs[1];

  if (data_output == nullptr || label_output == nullptr)
    return false;

  bool valid = inputs.size() == 0 && outputs.size() == 2 &&
               // Check data output
               data_output->data.samples() == batch_size_ &&
               data_output->data.width() == data_.width() &&
               data_output->data.height() == data_.height() &&
               data_output->data.maps() == data_.maps() &&
               // Check label output
               label_output->data.samples() == batch_size_ &&
               label_output->data.width() == labels_.width() &&
               label_output->data.height() == labels_.height() &&
               label_output->data.maps() == labels_.maps();

  if (valid) {
    data_output_ = data_output;
    label_output_ = label_output;
  }

  return valid;
}

void CVLabeledDataLayer::FeedForward() {
  for (std::size_t sample = 0; sample < batch_size_; sample++) {
    unsigned int selected_sample = 0;

    // Select samples until one from the right subset is hit
    do {
      // Select a sample from the permutation
      selected_sample = perm_[current_element_];

      // Select next element
      current_element_++;

      // If this is is out of bounds, start at the beginning and randomize
      // again.
      if (current_element_ >= perm_.size()) {
        current_element_ = 0;
        RedoPermutation();
      }
    } while (testing_ == (subset_[selected_sample] == testing_subset_));

    // Copy data
    Tensor::CopySample (data_, selected_sample, data_output_->data, sample);

    // Copy label
    Tensor::CopySample (labels_, selected_sample, label_output_->data, sample);
  }
}

void CVLabeledDataLayer::BackPropagate() {
  // No inputs, no backprop.
}

void CVLabeledDataLayer::SetCrossValidationSplit (const unsigned int split) {
  LOGDEBUG << "Selected " << split << "-fold cross-validation.";
  split_ = split;

  // Calculate subsets for each sample
  if (subset_ != nullptr)
    delete[] subset_;

  // Start random engine
  std::uniform_int_distribution<unsigned int> dist (0, split - 1);

  // Count samples
  std::size_t samples = data_.samples();

  // Allocate memory for subset_
  subset_ = new unsigned int [samples];

  // Assign subset to samples
  for (std::size_t i = 0; i < samples; i++) {
    subset_[i] = dist (generator_);
  }
}

void CVLabeledDataLayer::SetCrossValidationTestingSubset
(const unsigned int testing_subset) {
  LOGDEBUG << "Selected subset " << testing_subset << " for testing.";
  testing_subset_ = testing_subset;
}

void CVLabeledDataLayer::SetTestingMode (bool testing) {
  if (testing != testing_) {
    if (testing) {
      LOGDEBUG << "Enabled testing mode.";
    } else {
      LOGDEBUG << "Enabled training mode.";
    }
  }
  testing_ = testing;
}

void CVLabeledDataLayer::RedoPermutation() {
  // Shuffle the array
  std::shuffle (perm_.begin(), perm_.end(), generator_);
}

unsigned int CVLabeledDataLayer::GetSamplesInTrainingSet() {
  unsigned int count = 0;
  for (std::size_t i = 0; i < data_.samples(); i++) {
    if (subset_[i] != testing_subset_)
      count++;
  }
  return count;
}

unsigned int CVLabeledDataLayer::GetSamplesInTestingSet() {
  unsigned int count = 0;
  for (std::size_t i = 0; i < data_.samples(); i++) {
    if (subset_[i] == testing_subset_)
      count++;
  }
  return count;
}

unsigned int CVLabeledDataLayer::GetBatchSize() {
  return batch_size_;
}


}
