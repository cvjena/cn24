/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <algorithm>

#include "SimpleLabeledDataLayer.h"

namespace Conv {

SimpleLabeledDataLayer::SimpleLabeledDataLayer (Conv::Tensor& training_data,
    Conv::Tensor& training_labels,
    Conv::Tensor& testing_data,
    Conv::Tensor& testing_labels,
    const unsigned int batch_size,
    const int seed) :
  training_data_ (std::move (training_data)),
  training_labels_ (std::move (training_labels)),
  testing_data_ (std::move (testing_data)),
  testing_labels_ (std::move (testing_labels)),
  batch_size_ (batch_size), seed_ (seed), generator_ (seed) {

  // Validate dataset
  if (training_data_.samples() != training_labels_.samples()) {
    FATAL ("Training data and labels don't have the same number of samples!");
  }

  if (testing_data_.samples() != testing_labels_.samples()) {
    FATAL ("Testing data and labels don't have the same number of samples!");
  }

  // Validate dimensions
  if (training_data_.width() != testing_data_.width() ||
    training_data_.height() != testing_data_.height() ||
    training_data_.maps() != testing_data_.maps() ||
    training_labels_.width() != testing_labels_.width() ||
    training_labels_.height() != testing_labels_.height() ||
    training_labels_.maps() != testing_labels_.maps()) {
    FATAL ("Dimensions don't agree");
  }

  LOGDEBUG << "Instance created. Training: " << training_data_ <<
           ", testing: " << testing_data_;

  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }
  
  if (training_data_.samples() < testing_data_.samples()) {
    LOGINFO << "WARNING: More testing data than training data!";
    LOGINFO << "WARNING: Please check your parameters";
  }

  // Generate random permutation of the samples
  // First, we need an array of ascending numbers
  for (unsigned int i = 0; i < training_data_.samples(); i++) {
    perm_training_.push_back (i);
  }
  for (unsigned int i = 0; i < testing_data_.samples(); i++) {
    perm_testing_.push_back (i);
  }

  RedoPermutationTraining();
  RedoPermutationTesting();
}

bool SimpleLabeledDataLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs) {
  if (inputs.size() != 0) {
    LOGERROR << "Inputs specified but not supported";
    return false;
  }

  CombinedTensor* data_output =
    new CombinedTensor (batch_size_, training_data_.width(),
                        training_data_.height(), training_data_.maps());

  CombinedTensor* label_output =
    new CombinedTensor (batch_size_, training_labels_.width(),
                        training_labels_.height(), training_labels_.maps());

  outputs.push_back (data_output);
  outputs.push_back (label_output);
  return true;
}

bool SimpleLabeledDataLayer::Connect (
  const std::vector< CombinedTensor* >& inputs,
  const std::vector< CombinedTensor* >& outputs) {
  CombinedTensor* data_output = outputs[0];
  CombinedTensor* label_output = outputs[1];

  if (data_output == nullptr || label_output == nullptr)
    return false;

  bool valid = inputs.size() == 0 && outputs.size() == 2 &&
               // Check data output
               data_output->data.samples() == batch_size_ &&
               data_output->data.width() == training_data_.width() &&
               data_output->data.height() == training_data_.height() &&
               data_output->data.maps() == training_data_.maps() &&
               // Check label output
               label_output->data.samples() == batch_size_ &&
               label_output->data.width() == training_labels_.width() &&
               label_output->data.height() == training_labels_.height() &&
               label_output->data.maps() == training_labels_.maps();

  if (valid) {
    data_output_ = data_output;
    label_output_ = label_output;
  }

  return valid;
}

void SimpleLabeledDataLayer::FeedForward() {
  for (std::size_t sample = 0; sample < batch_size_; sample++) {
    if (testing_) {
      // Select a sample from the permutation
      std::size_t selected_sample = perm_testing_[current_element_testing_];

      // Select next element
      current_element_testing_++;

      // If this is is out of bounds, start at the beginning and randomize
      // again.
      if (current_element_testing_ >= perm_testing_.size()) {
        current_element_testing_ = 0;
        RedoPermutationTesting();
      }
      // Copy data
      Tensor::CopySample (testing_data_, selected_sample, data_output_->data, sample);

      // Copy label
      Tensor::CopySample (testing_labels_, selected_sample, label_output_->data, sample);

    } else {
      // Select a sample from the permutation
      std::size_t selected_sample = perm_training_[current_element_training_];

      // Select next element
      current_element_training_++;

      // If this is is out of bounds, start at the beginning and randomize
      // again.
      if (current_element_training_ >= perm_training_.size()) {
        current_element_training_ = 0;
        RedoPermutationTraining();
      }
      // Copy data
      Tensor::CopySample (training_data_, selected_sample, data_output_->data, sample);

      // Copy label
      Tensor::CopySample (training_labels_, selected_sample, label_output_->data, sample);
    }


  }
}

void SimpleLabeledDataLayer::BackPropagate() {
  // No inputs, no backprop.
}

void SimpleLabeledDataLayer::SetTestingMode (bool testing) {
  if (testing != testing_) {
    if (testing) {
      LOGDEBUG << "Enabled testing mode.";
    } else {
      LOGDEBUG << "Enabled training mode.";
    }
  }
  testing_ = testing;
}

unsigned int SimpleLabeledDataLayer::GetSamplesInTrainingSet() {
  return training_data_.samples();
}

unsigned int SimpleLabeledDataLayer::GetSamplesInTestingSet() {
  return testing_data_.samples();
}

unsigned int SimpleLabeledDataLayer::GetBatchSize() {
  return batch_size_;
}


void SimpleLabeledDataLayer::RedoPermutationTraining() {
  // Shuffle the array
  std::shuffle (perm_training_.begin(), perm_training_.end(), generator_);
}

void SimpleLabeledDataLayer::RedoPermutationTesting() {
  // Shuffle the array
  std::shuffle (perm_testing_.begin(), perm_testing_.end(), generator_);
}


}
