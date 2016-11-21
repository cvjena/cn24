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
#include <cmath>

#include "NetGraph.h"
#include "StatAggregator.h"
#include "Init.h"
#include "DatasetInputLayer.h"

#define MAX_3(a,b,c) (a > b ? a : b) > c ? (a > b ? a : b) : c
#define MIN_3(a,b,c) (a < b ? a : b) < c ? (a < b ? a : b) : c
#define CLAMP(a) a < 0 ? 0 : (a > 1 ? 1 : a)

namespace Conv {

DatasetInputLayer::DatasetInputLayer (JSON configuration,
                                      Dataset* initial_dataset,
                                      const unsigned int batch_size,
                                      const datum loss_sampling_p,
                                      const unsigned int seed) :
  Layer(configuration),
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

  JSON_TRY_INT(flip_, configuration_, "flip", 0);
  JSON_TRY_DATUM(jitter_, configuration_, "jitter_factor", 0);
  JSON_TRY_DATUM(exposure_, configuration_, "exposure", 1);
  JSON_TRY_DATUM(saturation_, configuration_, "saturation", 1);
  do_augmentation_ = (flip_ > 0) || (jitter_ > 0) || (exposure_ > 1) || (saturation_ > 1);
  if(do_augmentation_) {
    LOGINFO << "Using data augmentation: ";
    if(flip_ > 0)
      LOGDEBUG << " - Flipping";
    if(jitter_ > 0)
      LOGDEBUG << " - Random scaling (" << jitter_ << ")";
    if(exposure_ > 1)
      LOGDEBUG << " - Random exposure (" << exposure_ << ")";
    if(saturation_ > 1)
      LOGDEBUG << " - Random saturation (" << saturation_ << ")";
  }
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

    if(do_augmentation_) {
      preaug_metadata_buffer_ = new DatasetMetadataPointer[batch_size_];
      preaug_data_buffer_.Resize(data_output->data);
      augmented_boxes_.resize(batch_size_);
      for(unsigned int sample = 0; sample < batch_size_; sample++) {
        metadata_buffer_[sample] = &(augmented_boxes_[sample]);
      }
    }
  }

  return valid;
}

void DatasetInputLayer::SelectAndLoadSamples() {
#ifdef BUILD_OPENCL
  data_output_->data.MoveToCPU (true);
  label_output_->data.MoveToCPU (true);
  localized_error_output_->data.MoveToCPU (true);
#endif

  // Augmentation randomizers
  std::uniform_real_distribution<datum> jitter_dist(- jitter_, jitter_);
  std::bernoulli_distribution binary_dist;

  for(unsigned int sample = 0; sample < batch_size_; sample++) {
    std::uniform_real_distribution<datum> exposure_dist(1.0, binary_dist(generator_) ? exposure_ : (datum)1.0 / exposure_);
    std::uniform_real_distribution<datum> saturation_dist(1.0, binary_dist(generator_) ? saturation_ : (datum) 1.0 / saturation_);
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

    // Generate scaling and transpose data
    const datum left_border = jitter_dist(generator_);
    const datum right_border = (datum)1.0 + jitter_dist(generator_);
    const datum x_scale = (right_border - left_border);
    const datum x_transpose_nrm = left_border;
    const datum x_transpose_img = left_border * (datum)(preaug_data_buffer_.width() - 1);

    const datum top_border = jitter_dist(generator_);
    const datum bottom_border = (datum)1.0 + jitter_dist(generator_);
    const datum y_scale = (bottom_border - top_border);
    const datum y_transpose_nrm = top_border;
    const datum y_transpose_img = top_border * (datum)(preaug_data_buffer_.height() - 1);

    const bool flip_horizontal = binary_dist(generator_);
    const datum flip_offset = data_output_->data.width() - 1;
    const datum box_offset = flip_offset/(datum)(data_output_->data.width());


    // Copy image and label
    bool success;

    if (testing_)
      success = dataset->GetTestingSample (data_output_->data, label_output_->data, helper_output_->data, localized_error_output_->data, sample, selected_element);
    else {
      if(do_augmentation_) {
        success = dataset->GetTrainingSample(preaug_data_buffer_, label_output_->data, helper_output_->data,
                                             localized_error_output_->data, sample, selected_element);
        LoadSampleAugmented(sample, x_scale, x_transpose_img, y_scale, y_transpose_img, flip_horizontal, flip_offset);

        if(data_output_->data.maps() == 3) {
          // HSV conversion and exposure / saturation adjustment
          const datum saturation_factor = saturation_dist(generator_);
          const datum exposure_factor = exposure_dist(generator_);
          AugmentInPlaceSatExp(sample, saturation_factor, exposure_factor);
        }

      } else {
        success = dataset->GetTrainingSample(data_output_->data, label_output_->data, helper_output_->data,
                                             localized_error_output_->data, sample, selected_element);
      }
    }

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
      else {
        if(do_augmentation_) {
          success = dataset->GetTrainingMetadata(preaug_metadata_buffer_, sample, selected_element);
          std::vector<BoundingBox>* preaug_sample_boxes = (std::vector<BoundingBox>*)preaug_metadata_buffer_[sample];
          augmented_boxes_[sample].clear();
          for(BoundingBox bbox : *preaug_sample_boxes) {
            if(flip_horizontal)
              bbox.x = box_offset - bbox.x;
            // Transform into pixel space
            bbox.x *= (datum)data_output_->data.width();
            bbox.y *= (datum)data_output_->data.height();
            bbox.x = (bbox.x - x_transpose_img) / x_scale;
            bbox.y = (bbox.y - y_transpose_img) / y_scale;

            // And back into normalized space
            bbox.x /= (datum)data_output_->data.width();
            bbox.y /= (datum)data_output_->data.height();

            // Apply scale to width and height
            bbox.w /= x_scale;
            bbox.h /= y_scale;

            // Drop boxes with CG outside the image
            if(bbox.x >= 0 && bbox.x <= 1 && bbox.y >= 0 && bbox.y <= 1)
              augmented_boxes_[sample].push_back(bbox);

            metadata_buffer_[sample] = &(augmented_boxes_[sample]);
          }
        } else {
          success = dataset->GetTrainingMetadata(metadata_buffer_, sample, selected_element);
        }
      }
    }

    if (!success) {
      FATAL ("Cannot load metadata from Dataset!");
    }

  }
}

void DatasetInputLayer::AugmentInPlaceSatExp(unsigned int sample, const datum saturation_factor,
                                             const datum exposure_factor) {
  for (unsigned int y = 0; y < data_output_->data.height(); y++) {
            for (unsigned int x = 0; x < data_output_->data.width(); x++) {
              // Convert RGB pixel to HSV pixel
              const datum R = *data_output_->data.data_ptr_const(x, y, 0, sample);
              const datum G = *data_output_->data.data_ptr_const(x, y, 1, sample);
              const datum B = *data_output_->data.data_ptr_const(x, y, 2, sample);

              const datum Cmax = MAX_3(R, G, B);
              const datum Cmin = MIN_3(R, G, B);
              const datum Delta = Cmax - Cmin;

              datum H = 0;
              datum S = 0;
              datum V = Cmax;
              if(Delta > 0) {
                S = Delta / Cmax;
                if(Cmax == R) {
                  H =(G - B) / Delta;
                } else if(Cmax == G) {
                  H = 2 + (B - R) / Delta;
                } else if(Cmax == B) {
                  H = 4 + (R - G) / Delta;
                }
                H *= 60;
                if(H < 0)
                  H += 360;
              }

              // Apply exposure and saturation scaling
              S = CLAMP(S * saturation_factor);
              V = CLAMP(V * exposure_factor);


              // Convert HSV back to RGB
              datum NR = 0;
              datum NG = 0;
              datum NB = 0;
              if(S == 0) {
                NR = NG = NB = V;
              } else {
                H /= 60;
                int i = (int) floor(H);
                const datum f = H - (datum) i;
                const datum p = V * ((datum) 1 - S);
                const datum q = V * ((datum) 1 - S * f);
                const datum t = V * ((datum) 1 - S * ((datum) 1 - f));

                switch (i) {
                  case 0:
                    NR = V;
                    NG = t;
                    NB = p;
                    break;
                  case 1:
                    NR = q;
                    NG = V;
                    NB = p;
                    break;
                  case 2:
                    NR = p;
                    NG = V;
                    NB = t;
                    break;
                  case 3:
                    NR = p;
                    NG = q;
                    NB = V;
                    break;
                  case 4:
                    NR = t;
                    NG = p;
                    NB = V;
                    break;
                  case 5:
                  default:
                    NR = V;
                    NG = p;
                    NB = q;
                    break;
                }
              }

              *data_output_->data.data_ptr(x, y, 0, sample) = CLAMP(NR);
              *data_output_->data.data_ptr(x, y, 1, sample) = CLAMP(NG);
              *data_output_->data.data_ptr(x, y, 2, sample) = CLAMP(NB);
            }
          }
}

void DatasetInputLayer::LoadSampleAugmented(unsigned int sample, const datum x_scale, const datum x_transpose_img,
                                            const datum y_scale, const datum y_transpose_img,
                                            const bool flip_horizontal, const datum flip_offset) {
  for(unsigned int map = 0; map < data_output_->data.maps(); map++) {
#pragma omp parallel for default(shared)
          for(unsigned int y = 0; y < data_output_->data.height(); y++) {

            const datum origin_y = ((datum)y) * y_scale + y_transpose_img;
            if(origin_y >= 0 && origin_y <= (preaug_data_buffer_.height() - 1)) {

              for (unsigned int x = 0; x < data_output_->data.width(); x++) {
                const datum inner_x = (datum)x;
                const datum origin_x = flip_horizontal ? flip_offset - (inner_x * x_scale + x_transpose_img) : inner_x * x_scale + x_transpose_img;

                if(origin_x >= 0 && origin_x <= (preaug_data_buffer_.width() - 1)) {
                  *data_output_->data.data_ptr(x, y, map, sample) =
                      preaug_data_buffer_.GetSmoothData(origin_x, origin_y, map, sample);
                } else {
                  *data_output_->data.data_ptr(x, y, map, sample) = 0;
                }
              }
            } else {
              for (unsigned int x = 0; x < data_output_->data.width(); x++) {
                *data_output_->data.data_ptr(x, y, map, sample) = 0;
              }
            }
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
