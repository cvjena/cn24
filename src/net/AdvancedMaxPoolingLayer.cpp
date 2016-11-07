/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <limits>

#include "Log.h"
#include "CLHelper.h"
#include "AdvancedMaxPoolingLayer.h"
#include "ConfigParsing.h"

#ifdef BUILD_OPENCL
#define BUILD_OPENCL_MAX
#endif

namespace Conv {

AdvancedMaxPoolingLayer::AdvancedMaxPoolingLayer (const unsigned int region_width,
                                  const unsigned int region_height,
                                  const unsigned int stride_width,
                                  const unsigned int stride_height ) :
  SimpleLayer(JSON::object()),
  region_width_ (region_width), region_height_ (region_height),
  stride_width_ (stride_width), stride_height_ (stride_height){
  LOGDEBUG << "Instance created: " << region_width_ << "x" << region_height_ <<
           " pooling.";
}

AdvancedMaxPoolingLayer::AdvancedMaxPoolingLayer(JSON configuration) : SimpleLayer(configuration) {
  region_width_ = 1;
  region_height_ = 1;
  stride_width_ = 1;
  stride_height_ = 1;
  
	if(configuration.count("size") != 1 || !configuration["size"].is_array() || configuration["size"].size() != 2) {
		FATAL("Invalid configuration (no size): " << configuration.dump());
	} else {
		region_width_ = configuration["size"][0];
		region_height_ = configuration["size"][0];
	}
	
  stride_width_ = region_width_;
  stride_height_ = region_height_;
	
	if(configuration.count("stride") == 1 && configuration["stride"].is_array() && configuration["stride"].size() == 2) {
		stride_width_ = configuration["stride"][0];
		stride_height_ = configuration["stride"][1];
	}
	
	// TODO Validation of actual values
}

bool AdvancedMaxPoolingLayer::CreateOutputs (
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

  const int output_width = ((int)input->data.width() - (int)region_width_ ) / (int)stride_width_ + 1;
  const int output_height = ((int)input->data.height() - (int)region_height_) / (int)stride_height_ + 1;

  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      output_width, output_height,
      input->data.maps());

  // Tell network about the output
  outputs.push_back (output);

  return true;
}
  
bool AdvancedMaxPoolingLayer::Connect (const CombinedTensor* input,
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

  maximum_mask_.Resize (input->data.samples(), output_width_,
			output_height_, maps_);

  return true;
}

void AdvancedMaxPoolingLayer::FeedForward() {
#ifdef BUILD_OPENCL_MAX
  input_->data.MoveToGPU();
  output_->data.MoveToGPU(true);
  maximum_mask_.MoveToGPU(true);
  
  cl_uint error = 0;
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 0, sizeof (cl_mem), &input_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 1, sizeof (cl_mem), &maximum_mask_.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 2, sizeof (cl_mem), &output_->data.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 3, sizeof (unsigned int), &input_width_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 4, sizeof (unsigned int), &input_height_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 5, sizeof (unsigned int), &maps_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 6, sizeof (unsigned int), &output_width_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 7, sizeof (unsigned int), &output_height_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 8, sizeof (unsigned int), &region_width_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 9, sizeof (unsigned int), &region_height_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 10, sizeof (unsigned int), &stride_width_);
  error |= clSetKernelArg (CLHelper::k_amaximumForward, 11, sizeof (unsigned int), &stride_height_);
  if (error != CL_SUCCESS) {
    FATAL ("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = { output_width_, output_height_, maps_* input_->data.samples() };

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_amaximumForward, 3, NULL,
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
          for (unsigned int iy = oy * stride_height_;
                iy < (oy * stride_height_) + region_height_; iy++) {
            for (unsigned int ix = ox * stride_width_;
                ix < (ox * stride_width_) + region_width_; ix++) {
              const datum ival =
                *input_->data.data_ptr_const (ix, iy, map, sample);
              if (ival > maximum) {
                maximum = ival;
                mix = ix;
                miy = iy;
              }
            }
          }
          
          // Found maximum, save
          *maximum_mask_.data_ptr(ox, oy, map, sample) = input_width_ * miy + mix;
          
          // Feed forward
          *output_->data.data_ptr(ox, oy, map, sample) = maximum;
        }
      }
    }
  }
#endif
}

void AdvancedMaxPoolingLayer::BackPropagate() {
#ifdef BUILD_OPENCL_MAX
  input_->delta.MoveToGPU(true);
  output_->delta.MoveToGPU();
  maximum_mask_.MoveToGPU();
  
  cl_uint error = 0;
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 0, sizeof (cl_mem), &input_->delta.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 1, sizeof (cl_mem), &maximum_mask_.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 2, sizeof (cl_mem), &output_->delta.cl_data_ptr_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 3, sizeof (unsigned int), &input_width_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 4, sizeof (unsigned int), &input_height_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 5, sizeof (unsigned int), &maps_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 6, sizeof (unsigned int), &output_width_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 7, sizeof (unsigned int), &output_height_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 8, sizeof (unsigned int), &region_width_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 9, sizeof (unsigned int), &region_height_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 10, sizeof (unsigned int), &stride_width_);
  error |= clSetKernelArg (CLHelper::k_amaximumBackward, 11, sizeof (unsigned int), &stride_height_);
  if (error != CL_SUCCESS) {
    FATAL ("Error setting kernel args: " << (signed int) error);
  }

  size_t global_work_size[] = { input_width_, input_height_, maps_* input_->data.samples() };

  error = clEnqueueNDRangeKernel (CLHelper::queue, CLHelper::k_amaximumBackward, 3, NULL,
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
  
#define MP_HELPER_MIN(X, Y) (((X) < (Y)) ? (X) : (Y))
  
#pragma omp parallel for default(shared)
  for(std::size_t sample = 0; sample < input_->data.samples(); sample++) {
    for (unsigned int map = 0; map < maps_; map++) {
      for (unsigned int ix = 0; ix < input_width_; ix++) {
        for(unsigned int iy = 0; iy < input_height_; iy++) {
          const unsigned int mask_index = ix + input_width_ * iy;
          const unsigned int oxstart = (ix < region_width_) ? 
            0 : (ix - region_width_) / stride_width_+ 1;
          const unsigned int oxend = MP_HELPER_MIN(ix / stride_width_ + 1, output_width_);
          
          const unsigned int oystart = (iy < region_height_) ? 
            0 : (iy - region_height_) / stride_height_ + 1;
          const unsigned int oyend = MP_HELPER_MIN(iy / stride_height_ + 1, output_height_);
          
          datum sum = 0.0;
          for (unsigned int oy = oystart; oy < oyend; oy++) {
            for (unsigned int ox = oxstart; ox < oxend; ox++) {
              if(*maximum_mask_.data_ptr_const(ox, oy, map, sample) == mask_index)
                sum += *output_->delta.data_ptr_const(ox, oy, map, sample);
            }
          }
          *(input_->delta.data_ptr(ix, iy, map, sample)) = sum;
        }
      }
    }
  }
#endif
}


bool AdvancedMaxPoolingLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL_MAX
  return true;
#else
  return false;
#endif
}

}
