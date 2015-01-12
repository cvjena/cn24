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
#include <cstring>

#include "Log.h"

#include "Tensor.h"
#include "CombinedTensor.h"

#include "S2CVLabeledDataLayer.h"

namespace Conv {

S2CVLabeledDataLayer::S2CVLabeledDataLayer (Tensor& data, Tensor& labels,
    const unsigned int patchsize_x,
    const unsigned int patchsize_y,
    const unsigned int batch_size,
    const int seed,
    const unsigned int split,
    const unsigned int resume_from,
    const bool helper_position,
    const bool randomize_subsets,
    const bool normalize_mean,
    const bool normalize_stddev,
    const localized_error_function error_function
                                           ) :
  data_ (std::move (data)), labels_ (std::move (labels)),
  patchsize_x_ (patchsize_x), patchsize_y_ (patchsize_y),
  batch_size_ (batch_size), randomize_subsets_ (randomize_subsets),
  helper_x_ (helper_position), helper_y_ (helper_position),
  normalize_mean_ (normalize_mean), normalize_stddev_ (normalize_stddev),
  seed_ (seed), generator_ (seed), current_element_ (resume_from),
  error_function_ (error_function) {
  // Check if sample count matches
  if (data_.samples() != labels_.samples()) {
    FATAL ("The number of samples don't match. data: " << data_ <<
           ", labels: " << labels_);
  }

  LOGDEBUG << "Instance created: " << data_ << ", " << labels_;
  LOGDEBUG << "Using patch size: " << patchsize_x_ << " x " << patchsize_y_;

  if (helper_x_)
    LOGDEBUG << "Helper X enabled";
  if (helper_y_)
    LOGDEBUG << "Helper Y enabled";

  if (randomize_subsets)
    LOGDEBUG << "Randomizing subsets";
  else
    LOGDEBUG << "Not randomizing subsets";

  if (normalize_mean_)
    LOGDEBUG << "Mean normalization enabled";
  else
    LOGDEBUG << "Mean normalization disabled";

  if (normalize_stddev_)
    LOGDEBUG << "Standard deviation normalization enabled";
  else
    LOGDEBUG << "Standard deviation normalization disabled";

  if (normalize_stddev && !normalize_mean)
    FATAL ("Cannot normalize standard deviation without mean normalization!");

  if (error_function_ != DefaultLocalizedErrorFunction)
    LOGDEBUG << "Using custom error function";

  ppr_ = (data_.width() - (patchsize_x_ - 1));
  ppi_ =  ppr_ * (data_.height() - (patchsize_y_ - 1));
  patches_ = ppi_ * data_.samples();

  LOGDEBUG << "This will create " << patches_ << " patches...";

  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }

  // Generate random permutation of the samples
  // First, we need an array of ascending numbers
  for (unsigned int i = 0; i < patches_; i++) {
    perm_.push_back (i);
  }


  RedoPermutation();

  SetCrossValidationSplit (split);
  SetCrossValidationTestingSubset (0);
}


bool S2CVLabeledDataLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs) {
  if (inputs.size() != 0) {
    LOGERROR << "Inputs specified but not supported";
    return false;
  }

  CombinedTensor* data_output =
    new CombinedTensor (batch_size_, patchsize_x_,
                        patchsize_y_, data_.maps());

  CombinedTensor* label_output =
    new CombinedTensor (batch_size_, labels_.maps());

  CombinedTensor* helper_output =
    new CombinedTensor (
    batch_size_, (helper_x_ ? 1 : 0) + (helper_y_ ? 1 : 0));

  CombinedTensor* localized_error_output =
    new CombinedTensor (batch_size_);

  outputs.push_back (data_output);
  outputs.push_back (label_output);
  outputs.push_back (helper_output);
  outputs.push_back (localized_error_output);
  return true;
}

bool S2CVLabeledDataLayer::Connect (
  const std::vector< CombinedTensor* >& inputs,
  const std::vector< CombinedTensor* >& outputs) {
  // TODO validate
  CombinedTensor* data_output = outputs[0];
  CombinedTensor* label_output = outputs[1];
  CombinedTensor* helper_output = outputs[2];
  CombinedTensor* localized_error_output = outputs[3];

  if (data_output == nullptr || label_output == nullptr ||
      localized_error_output == nullptr)
    return false;

  bool valid = inputs.size() == 0 && outputs.size() == 4 &&
               // Check data output
               data_output->data.samples() == batch_size_ &&
               data_output->data.width() == patchsize_x_ &&
               data_output->data.height() == patchsize_y_ &&
               data_output->data.maps() == data_.maps() &&
               // Check label output
               label_output->data.samples() == batch_size_ &&
               label_output->data.width() == labels_.maps() &&
               label_output->data.height() == 1 &&
               label_output->data.maps() == 1 &&
               // Check helper output
               helper_output->data.samples() == batch_size_ &&
               helper_output->data.width() ==
               (helper_x_ ? 1 : 0) + (helper_y_ ? 1 : 0) &&
               helper_output->data.height() == 1 &&
               helper_output->data.maps() == 1 &&
               // Check localized error output
               localized_error_output->data.samples() == batch_size_ &&
               localized_error_output->data.width() == 1 &&
               localized_error_output->data.height() == 1 &&
               localized_error_output->data.maps() == 1;

  if (valid) {
    data_output_ = data_output;
    label_output_ = label_output;
    helper_output_ = helper_output;
    localized_error_output_ = localized_error_output;
  }

  return valid;
}

void S2CVLabeledDataLayer::FeedForward() {
  for (std::size_t sample = 0; sample < batch_size_; sample++) {
    unsigned int selected_sample = 0;

    if (testing_) {
      do {
        // The testing samples are not randomized
        selected_sample = current_element_testing_;
        current_element_testing_++;
        if (current_element_ >= patches_)
          current_element_testing_ = 0;
      } while (subset_[selected_sample] != testing_subset_);
    } else {

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
      } while ( (subset_[selected_sample] == testing_subset_)
                && !select_all_);
    }

    // Copy data
    unsigned int image = selected_sample / ppi_;
    unsigned int image_begin = image * ppi_;
    unsigned int image_y = (selected_sample - image_begin) / ppr_;
    unsigned int image_row_begin = image_begin + (image_y * ppr_);
    unsigned int image_x = (selected_sample - image_row_begin);

    for (unsigned int map = 0; map < data_.maps(); map++) {
      for (unsigned int y = 0; y < patchsize_y_; y++) {
        const datum* source = data_.data_ptr_const (image_x, image_y + y, map, image);
        datum* target = data_output_->data.data_ptr (0, y, map, sample);
        std::memcpy (target, source, sizeof (datum) * patchsize_x_ / sizeof (char));
      }
    }

    // Copy label
    for (unsigned int l = 0; l < labels_.maps(); l++) {
      datum* target = label_output_->data.data_ptr (l, 0, 0, sample);
      const datum label = *labels_.data_ptr_const (image_x + (patchsize_x_ / 2),
                          image_y + (patchsize_y_ / 2),
                          l, image);
      *target = label;
    }

    // Add helper output
    if (helper_y_ && !helper_x_)
      helper_output_->data[sample] =
        ( (datum) image_y) / ( (datum) data_.height());
    else if (helper_x_ && !helper_y_)
      helper_output_->data[sample] =
        ( (datum) image_x) / ( (datum) data_.width());
    else if (helper_x_ && helper_y_) {
      helper_output_->data[2 * sample] =
        ( (datum) image_y) / ( (datum) data_.height());
      helper_output_->data[2 * sample + 1] =
        ( (datum) image_x) / ( (datum) data_.width());
    }

    // Get localized error
    const datum localized_error = error_function_ (image_x + (patchsize_x_ / 2),
                                  image_y + (patchsize_y_ / 2));

    localized_error_output_->data[sample] = localized_error;
  }

  if (normalize_mean_) {
    unsigned int elements_per_sample = patchsize_x_ * patchsize_y_
                                       * data_.maps();
    #pragma omp parallel for default(shared)
    for (std::size_t sample = 0; sample < batch_size_; sample++) {
      // Add up elements
      datum sum = 0;
      for (unsigned int e = 0; e < elements_per_sample; e++) {
        sum += data_output_->data[sample * elements_per_sample + e];
      }

      // Calculate mean
      const datum mean = sum / (datum) elements_per_sample;

      // Substract mean
      for (unsigned int e = 0; e < elements_per_sample; e++) {
        data_output_->data[sample * elements_per_sample + e] -= mean;
      }

      if (normalize_stddev_) {
        // Calculate variance
        datum variance = 0;
        for (unsigned int e = 0; e < elements_per_sample; e++) {
          variance += pow (data_output_->data[sample * elements_per_sample + e]
                           , 2.0);
        }

        // Calculate standard deviation
        datum stddev = sqrt (variance);
        if (stddev == 0)
          stddev = 1;

        // Divide by standard deviation
        for (unsigned int e = 0; e < elements_per_sample; e++) {
          data_output_->data[sample * elements_per_sample + e] /= stddev;
        }

      } else {
      }
    }
  }
}

void S2CVLabeledDataLayer::BackPropagate() {
  // No inputs, no backprop.
}

void S2CVLabeledDataLayer::SetCrossValidationSplit (const unsigned int split) {
  LOGDEBUG << "Selected " << split << "-fold cross-validation.";
  split_ = split;

  // Calculate subsets for each sample
  if (subset_ != nullptr)
    delete[] subset_;

  // Allocate memory for subset_
  subset_ = new unsigned int [patches_];

  if (randomize_subsets_) {
    // Start random engine
    std::uniform_int_distribution<unsigned int> dist (0, split - 1);


    // Assign subset to samples
    for (std::size_t i = 0; i < patches_; i++) {
      subset_[i] = dist (generator_);
    }
  } else {
    // Not so random subsets (may be better for CV when using patches and
    // convolutional networks because the samples are more distinct.
    for (std::size_t i = 0; i < patches_; i++) {
      subset_[i] = (split * i) / patches_;
    }
  }
}

void S2CVLabeledDataLayer::SetCrossValidationTestingSubset
(const int testing_subset) {
  if (testing_subset == -1) {
    LOGDEBUG << "Selecting all subsets for training, none for testing.";
    select_all_ = true;
  } else if (testing_subset >= 0 && testing_subset < split_) {
    LOGDEBUG << "Selected subset " << testing_subset << " for testing.";
    testing_subset_ = testing_subset;
    select_all_ = false;
  } else {
    FATAL ("Subset out of bounds!");
  }
}

void S2CVLabeledDataLayer::SetTestingMode (bool testing) {
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

void S2CVLabeledDataLayer::RedoPermutation() {
  // Shuffle the array
  std::shuffle (perm_.begin(), perm_.end(), generator_);
}

unsigned int S2CVLabeledDataLayer::GetSamplesInTrainingSet() {
  unsigned int count = 0;
  for (std::size_t i = 0; i < patches_; i++) {
    if (subset_[i] != testing_subset_)
      count++;
  }
  return count;
}

unsigned int S2CVLabeledDataLayer::GetSamplesInTestingSet() {
  unsigned int count = 0;
  for (std::size_t i = 0; i < patches_; i++) {
    if (subset_[i] == testing_subset_)
      count++;
  }
  return count;
}

unsigned int S2CVLabeledDataLayer::GetBatchSize() {
  return batch_size_;
}


}

