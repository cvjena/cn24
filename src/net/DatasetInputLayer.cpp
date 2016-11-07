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
#include "StatAggregator.h"
#include "Init.h"
#include "DatasetInputLayer.h"

namespace Conv {

DatasetInputLayer::DatasetInputLayer (Dataset* initial_dataset,
                                      const unsigned int batch_size,
                                      const datum loss_sampling_p,
                                      const unsigned int seed) :
  Layer(JSON::object()),
  batch_size_ (batch_size),
  loss_sampling_p_ (loss_sampling_p),
  generator_ (seed), dist_ (0.0, 1.0) {
  LOGDEBUG << "Instance created.";

  label_maps_ = initial_dataset->GetLabelMaps();
  input_maps_ = initial_dataset->GetInputMaps();

  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }

  if(initial_dataset->GetMethod() == FCN && initial_dataset->GetTask() == SEMANTIC_SEGMENTATION) {
    LOGDEBUG << "Using loss sampling probability: " << loss_sampling_p_;
  } else {
    loss_sampling_p_ = 1.0;
  }

  AddDataset(initial_dataset, 1);

  SetActiveTestingDataset(initial_dataset);
}

void DatasetInputLayer::SetActiveTestingDataset(Dataset *dataset) {
  testing_dataset_ = dataset;
  LOGDEBUG << "Switching to testing dataset: " << dataset->GetName();
  elements_testing_ = dataset->GetTestingSamples();

  current_element_testing_ = 0;

  bool dataset_found = false;
  for(unsigned int i = 0; i < datasets_.size(); i++) {
    if(datasets_[i] == dataset) {
      System::stat_aggregator->SetCurrentTestingDataset(i);
      dataset_found = true;
      break;
    }
  }
  if(!dataset_found) {
    LOGERROR << "Testing dataset " << dataset->GetName() << " was not registered with DatasetInputLayer before!";
  }
}

bool DatasetInputLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                                       std::vector< CombinedTensor* >& outputs) {
  if (inputs.size() != 0) {
    LOGERROR << "Inputs specified but not supported";
    return false;
  }

  if(testing_dataset_->GetTask() == SEMANTIC_SEGMENTATION) {
    if (testing_dataset_->GetMethod() == FCN) {
      CombinedTensor *data_output =
          new CombinedTensor(batch_size_, testing_dataset_->GetWidth(),
                             testing_dataset_->GetHeight(), input_maps_);

      CombinedTensor *label_output =
          new CombinedTensor(batch_size_, testing_dataset_->GetWidth(),
                             testing_dataset_->GetHeight(), label_maps_);

      CombinedTensor *helper_output =
          new CombinedTensor(batch_size_, testing_dataset_->GetWidth(),
                             testing_dataset_->GetHeight(), 2);

      CombinedTensor *localized_error_output =
          new CombinedTensor(batch_size_, testing_dataset_->GetWidth(),
                             testing_dataset_->GetHeight(), 1);

      outputs.push_back(data_output);
      outputs.push_back(label_output);
      outputs.push_back(helper_output);
      outputs.push_back(localized_error_output);
    } else if (testing_dataset_->GetMethod() == PATCH) {
      CombinedTensor *data_output =
          new CombinedTensor(batch_size_, testing_dataset_->GetWidth(),
                             testing_dataset_->GetHeight(), input_maps_);

      CombinedTensor *label_output =
          new CombinedTensor(batch_size_, 1,
                             1, label_maps_);

      CombinedTensor *helper_output =
          new CombinedTensor(batch_size_, 1,
                             1, 2);

      CombinedTensor *localized_error_output =
          new CombinedTensor(batch_size_, 1,
                             1, 1);

      outputs.push_back(data_output);
      outputs.push_back(label_output);
      outputs.push_back(helper_output);
      outputs.push_back(localized_error_output);
    }
  } else if (testing_dataset_->GetTask() == CLASSIFICATION) {
    CombinedTensor *data_output =
        new CombinedTensor(batch_size_, testing_dataset_->GetWidth(),
                           testing_dataset_->GetHeight(), input_maps_);

    CombinedTensor *label_output =
        new CombinedTensor(batch_size_, 1,
                           1, label_maps_);

    CombinedTensor *helper_output =
        new CombinedTensor(batch_size_, 1,
                           1, 2);

    CombinedTensor *localized_error_output =
        new CombinedTensor(batch_size_, 1,
                           1, 1);

    outputs.push_back(data_output);
    outputs.push_back(label_output);
    outputs.push_back(helper_output);
    outputs.push_back(localized_error_output);
  } else if(testing_dataset_->GetTask() == DETECTION) {
    CombinedTensor *data_output =
        new CombinedTensor(batch_size_, testing_dataset_->GetWidth(),
                           testing_dataset_->GetHeight(), input_maps_);

    CombinedTensor *label_output =
        new CombinedTensor(batch_size_);

    CombinedTensor *helper_output =
        new CombinedTensor(batch_size_);

    CombinedTensor *localized_error_output =
        new CombinedTensor(batch_size_);

    DatasetMetadataPointer* metadata_buffer = new DatasetMetadataPointer[batch_size_];

    label_output->metadata = metadata_buffer;

    outputs.push_back(data_output);
    outputs.push_back(label_output);
    outputs.push_back(helper_output);
    outputs.push_back(localized_error_output);
  }
  return true;
}

bool DatasetInputLayer::Connect (const std::vector< CombinedTensor* >& inputs,
                                 const std::vector< CombinedTensor* >& outputs,
                                 const NetStatus* net) {
  UNREFERENCED_PARAMETER(net);
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
    if(testing_dataset_->GetTask() == DETECTION)
      metadata_buffer_ = label_output_->metadata;
  }

  return valid;
}

void DatasetInputLayer::SelectAndLoadSamples() {
#ifdef BUILD_OPENCL
  data_output_->data.MoveToCPU (true);
  label_output_->data.MoveToCPU (true);
  localized_error_output_->data.MoveToCPU (true);
#endif
  for(unsigned int sample = 0; sample < batch_size_; sample++) {
    unsigned int selected_element = 0;
    bool force_no_weight = false;
    Dataset* dataset = nullptr;

    if(testing_) {
      // No need to pick a dataset
      if(current_element_testing_ >= elements_testing_) {
        // Ignore this and further samples to avoid double testing some
        force_no_weight = true;
        selected_element = 0;
      } else {
        // Select the next testing element
        selected_element = current_element_testing_;
        current_element_testing_++;
      }
      dataset = testing_dataset_;
    } else {
      // Pick a dataset
      std::uniform_real_distribution<datum> dist(0.0, weight_sum_);
      datum selection = dist(generator_);
      for(unsigned int i=0; i < datasets_.size(); i++) {
        if(selection <= weights_[i]) {
          dataset = datasets_[i];
          break;
        }
        selection -= weights_[i];
      }
      if(dataset == nullptr) {
        FATAL("This can never happen. Dataset is null.");
      }

      // Pick an element
      std::uniform_int_distribution<unsigned int> element_dist(0, dataset->GetTrainingSamples() - 1);
      selected_element = element_dist(generator_);
    }

    // Copy image and label
    bool success;

    if (testing_)
      success = dataset->GetTestingSample (data_output_->data, label_output_->data, helper_output_->data, localized_error_output_->data, sample, selected_element);
    else
      success = dataset->GetTrainingSample (data_output_->data, label_output_->data, helper_output_->data, localized_error_output_->data, sample, selected_element);

    if (!success) {
      FATAL ("Cannot load samples from Dataset!");
    }

    // Perform loss sampling
    if (!testing_ && !force_no_weight && dataset->GetMethod() == FCN && dataset->GetTask() == SEMANTIC_SEGMENTATION) {
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

    // Clear localized error if possible
    if (force_no_weight)
      localized_error_output_->data.Clear (0.0, sample);

    // Load metadata
    if(dataset->GetTask() == DETECTION) {
      if (testing_)
        success = dataset->GetTestingMetadata(metadata_buffer_,sample, selected_element);
      else
        success = dataset->GetTrainingMetadata(metadata_buffer_,sample, selected_element);
    }

    if (!success) {
      FATAL ("Cannot load metadata from Dataset!");
    }

  }
}

void DatasetInputLayer::FeedForward() {
  // Nothing to do here
}

void DatasetInputLayer::BackPropagate() {
  // No inputs, no backprop.
}

unsigned int DatasetInputLayer::GetBatchSize() {
  return batch_size_;
}

unsigned int DatasetInputLayer::GetLabelWidth() {
  return (testing_dataset_->GetMethod() == PATCH || testing_dataset_->GetTask() == CLASSIFICATION || testing_dataset_->GetTask() == DETECTION) ? 1 : testing_dataset_->GetWidth();
}

unsigned int DatasetInputLayer::GetLabelHeight() {
  return (testing_dataset_->GetMethod() == PATCH || testing_dataset_->GetTask() == CLASSIFICATION || testing_dataset_->GetTask() == DETECTION) ? 1 : testing_dataset_->GetHeight();
}

unsigned int DatasetInputLayer::GetSamplesInTestingSet() {
  return testing_dataset_->GetTestingSamples();
}

unsigned int DatasetInputLayer::GetSamplesInTrainingSet() {
  unsigned int samples = 0;
  for(Dataset* dataset : datasets_)
    samples += dataset->GetTrainingSamples();
  return samples;
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

bool DatasetInputLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL
  return true;
#else
  return false;
#endif
}

void DatasetInputLayer::AddDataset(Dataset *dataset, const datum weight) {
  datasets_.push_back(dataset);
  weights_.push_back(weight);
  UpdateDatasets();
}

void DatasetInputLayer::SetWeight(Dataset *dataset, const datum weight) {
  // Find dataset
  bool found=false;
  for(unsigned int i=0; i < datasets_.size(); i++) {
    if(datasets_[i] == dataset) {
      weights_[i] = weight;
      LOGINFO << "Setting dataset \"" << dataset->GetName() << "\" weight: " << weight;
      found=true;
      break;
    }
  }

  if(!found) {
    FATAL("Could not find dataset \"" << dataset->GetName() << "\"");
  } else {
    UpdateDatasets();
  }
}

void DatasetInputLayer::UpdateDatasets() {
  // Calculate training elements
  elements_training_ = 0;
  weight_sum_ = 0;
  for(unsigned int i=0; i < datasets_.size(); i++) {
    elements_training_ += datasets_[i]->GetTrainingSamples();
    weight_sum_ += weights_[i];
  }
}

}
