/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

/**
* @file SegmentSetInputLayer.cpp
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
#include "SegmentSetInputLayer.h"

#define MAX_3(a,b,c) (a > b ? a : b) > c ? (a > b ? a : b) : c
#define MIN_3(a,b,c) (a < b ? a : b) < c ? (a < b ? a : b) : c
#define CLAMP(a) a < 0 ? 0 : (a > 1 ? 1 : a)

namespace Conv {

SegmentSetInputLayer::SegmentSetInputLayer (JSON configuration,
                                            Task task,
                                            ClassManager* class_manager,
                                      const unsigned int batch_size,
                                      const unsigned int seed) :
  Layer(configuration),
  class_manager_(class_manager),
  batch_size_ (batch_size),
  generator_ (seed), dist_ (0.0, 1.0), task_(task) {
  LOGDEBUG << "Instance created.";

  if(task == DETECTION) {
    input_maps_ = 3;
  } else {
    FATAL("NIY");
  }

  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }

  JSON_TRY_INT(input_width_, configuration_, "width", 448);
  JSON_TRY_INT(input_height_, configuration_, "height", 448);

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

  UpdateDatasets();
}

void SegmentSetInputLayer::SetActiveTestingSet(unsigned int index) {
  if(index < testing_sets_.size()) {
    testing_set_ = index;
    SegmentSet *set = testing_sets_[index];
    LOGDEBUG << "Switching to testing dataset: " << set->name;
    elements_testing_ = set->GetSampleCount();

    current_element_testing_ = 0;

    System::stat_aggregator->SetCurrentTestingDataset(index);
  } else {
    LOGERROR << "Testing dataset " << index << " out of bounds!";
  }
}

bool SegmentSetInputLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                                       std::vector< CombinedTensor* >& outputs) {
  if (inputs.size() != 0) {
    LOGERROR << "Inputs specified but not supported";
    return false;
  }

  if(task_ == SEMANTIC_SEGMENTATION) {
    FATAL("NIY")
  } else if (task_ == CLASSIFICATION) {
    FATAL("NIY");
  } else if(task_ == DETECTION) {
    CombinedTensor *data_output =
        new CombinedTensor(batch_size_, input_width_,
                           input_height_, input_maps_);

    CombinedTensor *label_output =
        new CombinedTensor(batch_size_);

    CombinedTensor *helper_output =
        new CombinedTensor(batch_size_);

    CombinedTensor *localized_error_output =
        new CombinedTensor(batch_size_);

    label_output->metadata = new DatasetMetadataPointer[batch_size_];

    outputs.push_back(data_output);
    outputs.push_back(label_output);
    outputs.push_back(helper_output);
    outputs.push_back(localized_error_output);
  }
  return true;
}

bool SegmentSetInputLayer::Connect (const std::vector< CombinedTensor* >& inputs,
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
    if(task_ == DETECTION) {
      metadata_.resize(batch_size_);
      for(unsigned int sample = 0; sample < batch_size_; sample++) {
        label_output_->metadata[sample] = &(metadata_[sample]);
      }
    }

    if(do_augmentation_) {
      preaug_data_buffer_.Resize(data_output->data);
      preaug_metadata_.resize(batch_size_);
    }
  }

  return valid;
}

bool SegmentSetInputLayer::ForceLoadDetection(JSON &sample, unsigned int index) {
#ifdef BUILD_OPENCL
  data_output_->data.MoveToCPU (true);
  localized_error_output_->data.MoveToCPU (true);
#endif
  localized_error_output_->data.Clear(1.0, index);
  return Segment::CopyDetectionSample(sample, index, &(data_output_->data), &(metadata_[index]), *class_manager_, Segment::SCALE);
}

void SegmentSetInputLayer::ForceWeightsZero() {
  for(unsigned int index = 0; index < localized_error_output_->data.samples(); index++) {
    localized_error_output_->data.Clear(0.0, index);
  }
}

void SegmentSetInputLayer::SelectAndLoadSamples() {
#ifdef BUILD_OPENCL
  data_output_->data.MoveToCPU (true);
  label_output_->data.MoveToCPU (true);
  localized_error_output_->data.MoveToCPU (true);
#endif

  // Augmentation randomizers
  std::uniform_real_distribution<datum> jitter_dist(- jitter_, jitter_);
  std::bernoulli_distribution binary_dist;

  for(unsigned int sample = 0; sample < batch_size_; sample++) {
    const datum rnd_exp = binary_dist(generator_) ? exposure_ : (datum)1.0 / exposure_;
    std::uniform_real_distribution<datum> exposure_dist(rnd_exp > 1 ? 1 : rnd_exp, rnd_exp > 1 ? rnd_exp : 1);
    const datum rnd_sat = binary_dist(generator_) ? saturation_ : (datum)1.0 / saturation_;
    std::uniform_real_distribution<datum> saturation_dist(rnd_sat > 1 ? 1: rnd_sat, rnd_sat > 1 ? rnd_sat : 1);
    unsigned int selected_element = 0;
    bool force_no_weight = false;
    SegmentSet* set = nullptr;

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
      set = testing_sets_[testing_set_];
    } else {
      // Pick a dataset
      std::uniform_real_distribution<datum> dist(0.0, training_weight_sum_);
      datum selection = dist(generator_);
      for(unsigned int i=0; i < training_sets_.size(); i++) {
        if(selection <= training_weights_[i]) {
          set = training_sets_[i];
          break;
        }
        selection -= training_weights_[i];
      }
      if(set == nullptr) {
        FATAL("This can never happen. Dataset is null.");
      }

      // Pick an element
      std::uniform_int_distribution<unsigned int> element_dist(0, set->GetSampleCount() - 1);
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

    if(do_augmentation_ && !testing_) {
      success = set->CopyDetectionSample(selected_element, sample, &(preaug_data_buffer_), &(preaug_metadata_[sample]), *class_manager_, Segment::SCALE);
      LoadSampleAugmented(sample, x_scale, x_transpose_img, y_scale, y_transpose_img, flip_horizontal, flip_offset);
      if(data_output_->data.maps() == 3) {
        // HSV conversion and exposure / saturation adjustment
        const datum saturation_factor = saturation_dist(generator_);
        const datum exposure_factor = exposure_dist(generator_);
        AugmentInPlaceSatExp(sample, saturation_factor, exposure_factor);
      }

      std::vector<BoundingBox>* preaug_sample_boxes = &(preaug_metadata_[sample]);
      metadata_[sample].clear();
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
        if(bbox.x >= 0 && bbox.x <= 1 && bbox.y >= 0 && bbox.y <= 1) {
          metadata_[sample].push_back(bbox);
        }
      }

    } else {
      success = set->CopyDetectionSample(selected_element, sample, &(data_output_->data), &(metadata_[sample]), *class_manager_, Segment::SCALE);
    }

    // Set weight tensor
    localized_error_output_->data.Clear(1.0, sample);

    if (!success) {
      FATAL ("Cannot load samples from Dataset!");
    }

    // Clear localized error if possible
    if (force_no_weight)
      localized_error_output_->data.Clear (0.0, sample);
  }
}

void SegmentSetInputLayer::AugmentInPlaceSatExp(unsigned int sample, const datum saturation_factor,
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

void SegmentSetInputLayer::LoadSampleAugmented(unsigned int sample, const datum x_scale, const datum x_transpose_img,
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

void SegmentSetInputLayer::FeedForward() {
  // Nothing to do here
}

void SegmentSetInputLayer::BackPropagate() {
  // No inputs, no backprop.
}

unsigned int SegmentSetInputLayer::GetBatchSize() {
  return batch_size_;
}

unsigned int SegmentSetInputLayer::GetLabelWidth() {
  return 1;
}

unsigned int SegmentSetInputLayer::GetLabelHeight() {
  return 1;
}

unsigned int SegmentSetInputLayer::GetSamplesInTestingSet() {
  return testing_sets_[testing_set_]->GetSampleCount();
}

unsigned int SegmentSetInputLayer::GetSamplesInTrainingSet() {
  unsigned int samples = 0;
  for(SegmentSet* segment_set : training_sets_)
    samples += segment_set->GetSampleCount();
  return samples;
}

void SegmentSetInputLayer::SetTestingMode (bool testing) {
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

void SegmentSetInputLayer::CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers) {
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

bool SegmentSetInputLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL
  return true;
#else
  return false;
#endif
}

void SegmentSetInputLayer::UpdateDatasets() {
  // Calculate training elements
  elements_training_ = 0;
  training_weight_sum_ = 0;
  for(unsigned int i=0; i < training_sets_.size(); i++) {
    elements_training_ += training_sets_[i]->GetSampleCount();
    training_weight_sum_ += training_weights_[i];
  }

  // Calculate testing elements
  elements_testing_ = 0;
  if(testing_sets_.size() > 0 && testing_set_ < testing_sets_.size()) {
    elements_testing_ = testing_sets_[testing_set_]->GetSampleCount();
  }
}

}
