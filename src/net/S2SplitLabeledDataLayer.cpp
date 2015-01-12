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

#include "S2SplitLabeledDataLayer.h"

namespace Conv {

S2SplitLabeledDataLayer::S2SplitLabeledDataLayer (
  std::istream& training, std::istream& testing,
  const unsigned int patchsize_x,
  const unsigned int patchsize_y,
  const unsigned int batch_size,
  const int seed,
  const unsigned int resume_from,
  const bool helper_position,
  const bool normalize_mean,
  const bool normalize_stddev,
  const int ignore_class,
  const localized_error_function error_function,
  const datum* per_class_weights
) :
  patchsize_x_ (patchsize_x), patchsize_y_ (patchsize_y),
  batch_size_ (batch_size), 
  helper_x_ (helper_position), helper_y_ (helper_position),
  normalize_mean_ (normalize_mean), normalize_stddev_ (normalize_stddev),
  ignore_class_(ignore_class), seed_ (seed), generator_ (seed), 
  current_element_ (resume_from), error_function_ (error_function),
  per_class_weights_(per_class_weights) 
{

  LOGDEBUG << "Instance created.";
  LOGDEBUG << "Using patch size: " << patchsize_x_ << " x " << patchsize_y_;

  if (helper_x_)
    LOGDEBUG << "Helper X enabled";
  if (helper_y_)
    LOGDEBUG << "Helper Y enabled";

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

  if (per_class_weights != nullptr)
    LOGDEBUG << "Using per-class weights";

  // Count tensors
  unsigned int tensor_count_training_ = 0;
  Tensor tensor;
  while(!training.eof()) {
    tensor.Deserialize(training);
    if(tensor.elements() == 0)
      break;
    // LOGDEBUG << "Tensor " << tensor_count_training_ << ": " << tensor;
    tensor_count_training_++;
  }
  LOGDEBUG << tensor_count_training_ << " training tensors";
  
  // We need alternating label and image tensors, so we need an even count
  if(tensor_count_training_ & 1) {
    FATAL("Odd training tensor count!");
  }
  
  unsigned int tensor_count_testing_ = 0;
  while(!testing.eof()) {
    tensor.Deserialize(testing);
    if(tensor.elements() == 0)
      break;
    // LOGDEBUG << "Tensor " << tensor_count_testing_ << ": " << tensor;
    tensor_count_testing_++;
  }
  LOGDEBUG << tensor_count_testing_ << " testing tensors";
  
  if(tensor_count_testing_ & 1) {
    FATAL("Odd testing tensor count!");
  }
  
  tensors_ = (tensor_count_testing_ + tensor_count_training_) / 2;
  
  // Reset streams
  training.clear();
  testing.clear();
  training.seekg(0, std::ios::beg);
  testing.seekg(0, std::ios::beg);
  
  // Allocate arrays that depend on the tensor count
  data_ = new Tensor[tensors_];
  labels_ = new Tensor[tensors_];
  first_element_ = new unsigned int[tensors_];
  last_element_ = new unsigned int[tensors_];
  
  // Read tensors
  unsigned int e = 0;
  for(unsigned int t = 0; t < (tensor_count_training_ / 2); t++) {
    data_[t].Deserialize(training);
    labels_[t].Deserialize(training);
    
    first_element_[t] = e;
    unsigned int count = ((data_[t].width() - (patchsize_x - 1))
    * (data_[t].height() - (patchsize_y - 1)));
    last_element_[t] = e + count - 1;
    
    e += count;
    elements_training_ += count;
  }
 
  for(unsigned int t = (tensor_count_training_ / 2) + 1; t < tensors_; t++) {
    data_[t].Deserialize(testing);
    labels_[t].Deserialize(testing);
    
    first_element_[t] = e;
    unsigned int count = ((data_[t].width() - (patchsize_x - 1))
    * (data_[t].height() - (patchsize_y - 1)));
    last_element_[t] = e + count - 1;
    
    e += count;
    elements_testing_ += count;
  }
  
  elements_total_ = elements_training_ + elements_testing_;
  input_maps_ = data_[0].maps();
  label_maps_ = labels_[0].maps();
  
  LOGDEBUG << "Training patches: " << elements_training_;
  LOGDEBUG << "Testing patches: " << elements_testing_;
  
  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }

  // Generate random permutation of the samples
  // First, we need an array of ascending numbers
  for (unsigned int i = 0; i < elements_training_; i++) {
    perm_.push_back (i);
  }


  RedoPermutation();
}


bool S2SplitLabeledDataLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs) {
  if (inputs.size() != 0) {
    LOGERROR << "Inputs specified but not supported";
    return false;
  }

  CombinedTensor* data_output =
    new CombinedTensor (batch_size_, patchsize_x_,
                        patchsize_y_, input_maps_);

  CombinedTensor* label_output =
    new CombinedTensor (batch_size_, label_maps_);

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

bool S2SplitLabeledDataLayer::Connect (
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
               data_output->data.maps() == input_maps_ &&
               // Check label output
               label_output->data.samples() == batch_size_ &&
               label_output->data.width() == label_maps_ &&
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

void S2SplitLabeledDataLayer::FeedForward() {
#ifdef BUILD_OPENCL
  data_output_->data.MoveToCPU(true);
  label_output_->data.MoveToCPU(true);
  if(helper_x_ || helper_y_)
    helper_output_->data.MoveToCPU(true);
  localized_error_output_->data.MoveToCPU(true);
#endif
  for (std::size_t sample = 0; sample < batch_size_; sample++) {
    unsigned int selected_element = 0;
    bool force_no_weight = false;
    if (testing_) {
      // The testing samples are not randomized
      selected_element = current_element_testing_;
      current_element_testing_++;
      if (current_element_testing_ >= elements_total_) {
        force_no_weight = true;
        selected_element = elements_training_ + 1;
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

    // Find out which tensor index belongs to the element in question
    bool element_found = false;
    unsigned int t;
    for (t = 0; t < tensors_; t++) {
      if (selected_element >= first_element_[t]
          && selected_element <= last_element_[t]) {
        element_found = true;
        break;
      }
    }
    
    if(!element_found) {
      FATAL("This patch does not exist, sorry!");
    }
    
    // Find out which patch needs to be extracted
    unsigned int w = data_[t].width();
    unsigned int ppr = (w - (patchsize_x_ - 1));
    
    unsigned int image_begin = first_element_[t];
    unsigned int image_y = (selected_element - image_begin) / ppr;
    unsigned int image_row_begin = image_begin + (image_y * ppr);
    unsigned int image_x = (selected_element - image_row_begin);

    for (unsigned int map = 0; map < input_maps_; map++) {
      for (unsigned int y = 0; y < patchsize_y_; y++) {
        const datum* source = data_[t].data_ptr_const (image_x, image_y + y, map);
        datum* target = data_output_->data.data_ptr (0, y, map, sample);
        std::memcpy (target, source, sizeof (datum) * patchsize_x_ / sizeof (char));
      }
    }

    // Copy label
    
    duint sample_class = 0;
    for (unsigned int l = 0; l < label_maps_; l++) {
      datum* target = label_output_->data.data_ptr (l, 0, 0, sample);
      const datum label = *(labels_[t].data_ptr_const (image_x + (patchsize_x_ / 2),
                          image_y + (patchsize_y_ / 2),
                          l));
      *target = label;
      sample_class = *((duint*)&label);
      if(sample_class == ignore_class_)
        force_no_weight = true;
    }

    // Add helper output
    if (helper_y_ && !helper_x_)
      helper_output_->data[sample] =
        ( (datum) image_y) / ( (datum) data_[t].height());
    else if (helper_x_ && !helper_y_)
      helper_output_->data[sample] =
        ( (datum) image_x) / ( (datum) data_[t].width());
    else if (helper_x_ && helper_y_) {
      helper_output_->data[2 * sample] =
        ( (datum) image_y) / ( (datum) data_[t].height());
      helper_output_->data[2 * sample + 1] =
        ( (datum) image_x) / ( (datum) data_[t].width());
    }

    // Get localized error
    const datum localized_error = error_function_ (image_x + (patchsize_x_ / 2),
                                  image_y + (patchsize_y_ / 2));

    localized_error_output_->data[sample] =
      force_no_weight ? 0.0 : localized_error;

    if(per_class_weights_ != nullptr)
      localized_error_output_->data[sample] *= per_class_weights_[sample_class];
  }

  if (normalize_mean_) {
    unsigned int elements_per_sample = patchsize_x_ * patchsize_y_
                                       * input_maps_;
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

void S2SplitLabeledDataLayer::BackPropagate() {
  // No inputs, no backprop.
}


void S2SplitLabeledDataLayer::SetTestingMode (bool testing) {
  if (testing != testing_) {
    if (testing) {
      LOGDEBUG << "Enabled testing mode.";

      // Always test the same elements for consistency
      current_element_testing_ = elements_training_ + 1;
    } else {
      LOGDEBUG << "Enabled training mode.";
    }
  }
  testing_ = testing;
}

void S2SplitLabeledDataLayer::RedoPermutation() {
  // Shuffle the array
  std::shuffle (perm_.begin(), perm_.end(), generator_);
}

unsigned int S2SplitLabeledDataLayer::GetSamplesInTrainingSet() {
  return elements_training_;
}

unsigned int S2SplitLabeledDataLayer::GetSamplesInTestingSet() {
  return elements_testing_;
}

unsigned int S2SplitLabeledDataLayer::GetBatchSize() {
  return batch_size_;
}


}

