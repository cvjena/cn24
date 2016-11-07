/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <limits>

#include "Log.h"
#include "CLHelper.h"
#include "MaxPoolingLayer.h"
#include "ConfigParsing.h"

#ifdef BUILD_OPENCL
#define BUILD_OPENCL_MAX
#endif

namespace Conv {

MaxPoolingLayer::MaxPoolingLayer (const unsigned int region_width,
                                  const unsigned int region_height) :
  SimpleLayer(JSON::object()),
  region_width_ (region_width), region_height_ (region_height) {
  LOGDEBUG << "Instance created: " << region_width_ << "x" << region_height_ <<
           " pooling.";
}

MaxPoolingLayer::MaxPoolingLayer(JSON configuration) :
  SimpleLayer(configuration) {
	region_width_ = 1;
  region_height_ = 1;
  
	if(configuration.count("size") != 1 || !configuration["size"].is_array() || configuration["size"].size() != 2) {
		FATAL("Invalid configuration (no size): " << configuration.dump());
	} else {
		region_width_ = configuration["size"][0];
		region_height_ = configuration["size"][0];
	}
	
	// TODO Validation of actual values	
}
  
bool MaxPoolingLayer::CreateOutputs (
  const std::vector< CombinedTensor* >& inputs,
  std::vector< CombinedTensor* >& outputs) {
  // This is a simple layer, only one input
  if (inputs.size() != 1) {
    LOGERROR << "Only one input supported!";
    return false;
  }

  // Save input node pointer
  CombinedTensor* input = inputs[0];

  // Check if input node pointer is null
  if (input == nullptr) {
    LOGERROR << "Null pointer input node!";
    return false;
  }

  // Validate dimensions
  if ( (input->data.width() % region_width_) != 0 ||
       (input->data.height() % region_height_) != 0) {
    LOGERROR << "Input dimensions not divisible by region dimensions!";
    return false;
  }

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      input->data.width() / region_width_, input->data.height() / region_height_,
      input->data.maps());

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool MaxPoolingLayer::Connect (const CombinedTensor* input,
                               CombinedTensor* output) {
  // TODO Validate dimensions
  bool valid = true;

  if (!valid) {
    LOGERROR << "Invalid dimensions!";
    return false;
  }

  // Save dimensions
  input_width_ = input->data.width();
  input_height_ = input->data.height();
  output_width_ = output->data.width();
  output_height_ = output->data.height();

  maps_ = input->data.maps();

#ifdef BUILD_OPENCL_MAX
  maximum_mask_.Resize (input->data.samples(), input_width_,
			input_height_, maps_);
#else
  // Create maximum Tensor
  maximum_ix_.Resize (input->data.samples(), output_width_,
                      output_height_, maps_);
  maximum_iy_.Resize (input->data.samples(), output_width_,
                      output_height_, maps_);
#endif

  return true;
}

void MaxPoolingLayer::FeedForward() {
#ifdef BUILD_OPENCL_MAX
  input_->data.MoveToGPU();
  output_->data.MoveToGPU(true);
  maximum_mask_.MoveToGPU(true);
  
  cl_uint error = 0;
  error |= clSetKernelArg (CLHelper::k_maximumForward, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 1, sizeof (cl_mem), &maximum_mask_.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 2, sizeof (cl_mem), &output_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 3, sizeof (unsigned int), &input_width_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 4, sizeof (unsigned int), &input_height_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 5, sizeof (unsigned int), &maps_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 6, sizeof (unsigned int), &output_width_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 7, sizeof (unsigned int), &output_height_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 8, sizeof (unsigned int), &region_width_);
  error |= clSetKernelArg (CLHelper::k_maximumForward, 9, sizeof (unsigned int), &region_height_);
  if (error != CL_SUCCESS) {
    FATAL ("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = { output_width_, output_height_, maps_* input_->data.samples() };

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_maximumForward, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    FATAL ("Error enqueueing kernel: " << (signed int) error);
  }

#ifdef BRUTAL_FINISH
  error = clFinish (CLHelper::queue);
  if (error != CL_SUCCESS) {
    FATAL ("Error finishing command queue: " << (signed int) error);
  }
#endif

#else
#pragma omp parallel for default(shared)
  for (std::size_t sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int map = 0; map < maps_; map++) {
      for (unsigned int ox = 0; ox < output_width_; ox++) {
        for (unsigned int oy = 0; oy < output_height_; oy++) {
          // Find maximum in region
          datum maximum = std::numeric_limits<datum>::lowest();
          unsigned int mix = 0;
          unsigned int miy = 0;
          for (unsigned int ix = ox * region_width_;
               ix < (ox + 1) * region_width_; ix++) {
            for (unsigned int iy = oy * region_height_;
                 iy < (oy + 1) *region_height_; iy++) {
              const datum ival =
                *input_->data.data_ptr_const (ix, iy, map, sample);
              if (ival > maximum) {
                maximum = ival;
                mix = ix;
                miy = iy;
		// mix = ox * region_width_;
		// miy = oy * region_height_;
              }
            }
          }
          
          // Found maximum, save
          *maximum_ix_.data_ptr(ox, oy, map, sample) = mix;
          *maximum_iy_.data_ptr(ox, oy, map, sample) = miy;
          
          // Feed forward
          *output_->data.data_ptr(ox, oy, map, sample) = maximum;
        }
      }
    }
  }
#endif
}

void MaxPoolingLayer::BackPropagate() {
#ifdef BUILD_OPENCL_MAX
  input_->delta.MoveToGPU(true);
  output_->delta.MoveToGPU();
  maximum_mask_.MoveToGPU();
  
  cl_uint error = 0;
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 0, sizeof (cl_mem), &input_->delta.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 1, sizeof (cl_mem), &maximum_mask_.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 2, sizeof (cl_mem), &output_->delta.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 3, sizeof (unsigned int), &input_width_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 4, sizeof (unsigned int), &input_height_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 5, sizeof (unsigned int), &maps_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 6, sizeof (unsigned int), &output_width_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 7, sizeof (unsigned int), &output_height_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 8, sizeof (unsigned int), &region_width_);
  error |= clSetKernelArg (CLHelper::k_maximumBackward, 9, sizeof (unsigned int), &region_height_);
  if (error != CL_SUCCESS) {
    FATAL ("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = { input_width_, input_height_, maps_* input_->data.samples() };

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_maximumBackward, 3, NULL,
                                  global_work_size, NULL, 0, NULL, NULL);
  if (error != CL_SUCCESS) {
    FATAL ("Error enqueueing kernel: " << (signed int) error);
  }

#ifdef BRUTAL_FINISH
  error = clFinish (CLHelper::queue);
  if (error != CL_SUCCESS) {
    FATAL ("Error finishing command queue: " << (signed int) error);
  }
#endif

#else
  input_->delta.Clear();
  
#pragma omp parallel for default(shared)
  for(std::size_t sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int map = 0; map < maps_; map++) {
      for (unsigned int ox = 0; ox < output_width_; ox++) {
        for (unsigned int oy = 0; oy < output_height_; oy++) {
          unsigned int ix = *maximum_ix_.data_ptr_const(ox, oy, map, sample);
          unsigned int iy = *maximum_iy_.data_ptr_const(ox, oy, map, sample);
          *(input_->delta.data_ptr(ix, iy, map, sample)) = 
             *output_->delta.data_ptr_const(ox, oy, map, sample);
        }
      }
    }
  }
  
  return;
  
#endif
}


bool MaxPoolingLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL_MAX
  return true;
#else
  return false;
#endif
}

}
