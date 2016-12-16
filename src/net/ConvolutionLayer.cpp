/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <cstring>
#include <algorithm>

#ifdef BUILD_OPENCL
#define BUILD_OPENCL_CONV
#endif

#ifdef BUILD_OPENCL_CONV
#include <iomanip>
#endif

#include "Config.h"
#include "Log.h"
#include "CLHelper.h"
#include "TensorMath.h"
#include "ConfigParsing.h"

#include "TensorViewer.h"

#include "ConvolutionLayer.h"

namespace Conv {

ConvolutionLayer::ConvolutionLayer (const unsigned int kwidth,
                                    const unsigned int kheight,
                                    const unsigned int output_maps,
                                    const unsigned int stride_width,
                                    const unsigned int stride_height,
                                    const unsigned int pad_width,
                                    const unsigned int pad_height,
                                    const unsigned int group,
                                    const int seed, const datum dropout_fraction) :
  SimpleLayer(JSON::object()),
  output_maps_ (output_maps), kernel_width_ (kwidth), kernel_height_ (kheight),
  rand_ (seed), stride_width_(stride_width), stride_height_(stride_height),
  pad_width_(pad_width), pad_height_(pad_height),
  group_(group), dropout_fraction_(dropout_fraction) {
  // Validate kernel dimensions. These are very important because the
  // FeedForward and BackPropagate implementations rely on some assumptions.

  // The kernels must not be of zero size in any dimension.
  if (kernel_width_ == 0 || kernel_height_ == 0) {
    FATAL ("Kernels cannot have zero dimensions");
  }
    
  if(stride_width_ == 0 || stride_height_ == 0) {
    FATAL("Stride needs to be at least 1!");
  }
  
  if((output_maps_ % group) != 0) {
    FATAL("Output maps need to divide group count");
  }
  
  // Using a zero seed on more than one layer may introduce symmetries and
  // influence the gain of a network in a negative way
  if (seed == 0) {
    LOGWARN << "Random seed is zero";
  }

  LOGDEBUG << "Instance created. " << output_maps_ << " output maps with " <<
           kernel_width_ << "x" << kernel_height_ << " kernels, stride: "<<  stride_width << "x" << stride_height_ << 
           ", padding: " << pad_width_ << "x" << pad_height_ << ", group: " << group_ << ".";
  LOGDEBUG << "Dropout fraction: " << dropout_fraction_;
}

ConvolutionLayer::ConvolutionLayer(JSON configuration) :
  SimpleLayer(configuration) {
  unsigned int seed = 0;
  kernel_width_ = 0;
  kernel_height_ = 0;
  output_maps_ = 0;
  stride_width_ = 1;
  stride_height_ = 1;
  pad_width_ = 0;
  pad_height_ = 0;
  group_ = 1;
  dropout_fraction_ = 0.0;
  datum local_lr = 1.0;
	
	if(configuration.count("size") != 1 || !configuration["size"].is_array() || configuration["size"].size() != 2) {
		FATAL("Invalid configuration (no size): " << configuration.dump());
	} else {
		kernel_width_ = configuration["size"][0];
		kernel_height_ = configuration["size"][1];
	}
	
	if(configuration.count("stride") == 1 && configuration["stride"].is_array() && configuration["stride"].size() == 2) {
		stride_width_ = configuration["stride"][0];
		stride_height_ = configuration["stride"][1];
	}
  
	if(configuration.count("pad") == 1 && configuration["pad"].is_array() && configuration["pad"].size() == 2) {
		pad_width_ = configuration["pad"][0];
		pad_height_ = configuration["pad"][1];
	}
	
	if(configuration.count("kernels") == 1 && configuration["kernels"].is_number()) {
		output_maps_ = configuration["kernels"];
	}
  
	if(configuration.count("group") == 1 && configuration["group"].is_number()) {
		group_ = configuration["group"];
	}
  
	if(configuration.count("dropout") == 1 && configuration["dropout"].is_number()) {
		dropout_fraction_ = configuration["dropout"];
	}
	
	if(configuration.count("llr") == 1 && configuration["llr"].is_number()) {
		local_lr = configuration["llr"];
	}
  
	if(configuration.count("seed") == 1 && configuration["seed"].is_number()) {
		seed = configuration["seed"];
	}
  
  rand_.seed(seed);
  SetLocalLearningRate(local_lr);
	
  // TODO Validation like in large constructor
}

bool ConvolutionLayer::CreateOutputs (
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
  // The input maps have to be larger or at least as large as the kernel,
  // because we only do 'valid' convolutions.
  // For MNIST recognition, LeCun says that 'same' convolutions perform better
  // as a first layer. He adds a border around the training and test sets to
  // achieve the same thing without changing the code.
  const int output_width = ((int)pad_width_ + (int)pad_width_ + (int)input->data.width() - (int)kernel_width_) / (int)stride_width_ + 1;
  const int output_height = ((int)pad_height_ + (int)pad_height_ + (int)input->data.height() - (int)kernel_height_) / (int)stride_height_ + 1;
  
  if (output_width <= 0 || output_height <= 0) {
    LOGERROR << "Unsupported input dimensions " << input->data;
    return false;
  }
  
  if ((input->data.maps() % group_) != 0) {
    FATAL("Input maps need to divide group count!");
  }
  
  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
      output_width, output_height, output_maps_);

  // Tell network about the output
  outputs.push_back (output);

  return true;
}

bool ConvolutionLayer::Connect (const CombinedTensor* input,
                                CombinedTensor* output) {
  bool valid =
    // input->data.width() >= kernel_width_ && input->data.height() >=  kernel_height_ &&
    output->data.width() == (pad_width_ + pad_width_ + input->data.width() - kernel_width_) / stride_width_  + 1 &&
    output->data.height() == (pad_height_ + pad_height_ + input->data.height() - kernel_height_) / stride_height_ + 1;

  if (!valid) {
    return false;
  }

  // Save parameters
  input_maps_ = input->data.maps();
  input_width_ = input->data.width();
  input_height_ = input->data.height();
  output_width_ = output->data.width();
  output_height_ = output->data.height();

  LOGDEBUG << "Local learning rate is now " << local_lr_;
  
  // Create im2col output buffer
  im2col_ff_buffer.Resize (kernel_width_ * kernel_height_ * input_maps_, output_width_,
                           output_height_, input->data.samples());
  
  sms_ff_buffer.Resize(output_maps_, output_width_, output_height_, input->data.samples());
  
  sms2_bp_buffer.Resize(output_maps_, output_width_, output_height_, input->data.samples());

  bp_deltax_buffer.Resize (kernel_width_ * kernel_height_ * input_maps_, output_width_,
                           output_height_, input->data.samples());

  // This is faster than adding manually...
  ones_.Resize (1, output_width_ * output_height_ * input->data.samples());

  for (unsigned int i = 0; i < ones_.elements(); i++) {
    ones_[i] = 1;
  }

  // Create kernels
  weights_ = new CombinedTensor (output_maps_, kernel_width_, kernel_height_, input_maps_ / group_);
  bias_ = new CombinedTensor (1, output_maps_);

  // Initialize weights to zero so the net won't work if Net::InitializeWeights
  // is not called. Random memory junk may work but is certainly not optimal.
  bias_->data.Clear();
  weights_->data.Clear();
  
  // Initialize the dropout mask tensor
  dropout_mask_.Resize(input->data.samples(),output_maps_);

  // Tell the net about our parameters
  parameters_.push_back (weights_);
  parameters_.push_back (bias_);

  return true;
}

void ConvolutionLayer::FeedForward() {
  const datum p = net_->IsTesting() ? 0.0 : dropout_fraction_;
  const datum w = net_->IsTesting() ? (1.0 - dropout_fraction_) : 1.0;
  
  im2col_ff_buffer.hint_ignore_content_ = true;
  output_->data.hint_ignore_content_ = true;
  sms_ff_buffer.hint_ignore_content_ = true;
  
  TensorMath::IM2COL(input_->data, input_width_, input_height_, input_maps_, input_->data.samples(),
        kernel_width_, kernel_height_, stride_width_, stride_height_, pad_width_, pad_height_, im2col_ff_buffer);
  

  for(unsigned int g = 0; g < group_; g++) {
    // Convolve
    TensorMath::GEMM(true, false, false, output_maps_ / group_,
          output_width_ * output_height_ * input_->data.samples(),
          (kernel_width_ * kernel_height_ * input_maps_) / group_,
          w, weights_->data, (g * output_maps_) / group_, (kernel_width_ * kernel_height_ * input_maps_) / group_,
          im2col_ff_buffer, (kernel_width_ * kernel_height_ * input_maps_ * g) / group_, output_width_ * output_height_ * input_->data.samples(),
          0.0, sms_ff_buffer, (g * output_maps_) / group_, output_width_ * output_height_ * input_->data.samples());
  }
  
  // Add bias
  TensorMath::GEMM (true, false, false, output_maps_,
        output_width_ * output_height_ * input_->data.samples(), 1, w, bias_->data, 0, 1,
        ones_, 0, output_width_ * output_height_ * input_->data.samples(),
        1.0, sms_ff_buffer, 0, output_width_ * output_height_ * input_->data.samples());

  TensorMath::SMS(sms_ff_buffer, output_->data);

  /*for(unsigned int sample = 0; sample < input_->data.samples(); sample++) {
    // Add bias
    TensorMath::GEMM (true, false, false, output_maps_,
          output_width_ * output_height_, 1, w, bias_->data, 0, 1,
          ones_, 0, output_width_ * output_height_,
          1.0, output_->data, sample, output_width_ * output_height_);

  }*/

  // Very simple dropout FF implementation
  // This could be optimized a _lot_
  if(p == 0.0) {
    //dropout_mask_.Clear(1.0);
  } else {
    FATAL("Dropout is not yet TensorMath compatible");
  /*
    std::uniform_real_distribution<datum> dist(0.0, 1.0);
    for(unsigned int e = 0; e < input_->data.samples() * output_maps_; e++) {
      dropout_mask_[e] = dist(rand_) < p ? 0.0 : 1.0;
    }
  }
  
  unsigned int sk_id = 0;
  for(unsigned int s = 0; s < input_->data.samples(); s++) {
    for(unsigned int m = 0; m < output_maps_; m++) {
      datum* target_map = output_->data.data_ptr(0,0,m,s);
      for(unsigned int e = 0; e < output_width_ * output_height_; e++) {
        if(dropout_mask_[sk_id] == 0.0)
          target_map[e] = 0.0;
      }
    }
    sk_id++;*/
  }
}

void ConvolutionLayer::BackPropagate() {
  // Very simple dropout backprop implementation
  // This could be optimized a _lot_
  /*unsigned int sk_id = 0;
  for(unsigned int s = 0; s < input_->data.samples(); s++) {
    for(unsigned int m = 0; m < output_maps_; m++) {
      datum* target_map = output_->delta.data_ptr(0,0,m,s);
      if(dropout_mask_[sk_id] == 0.0)
        for(unsigned int e = 0; e < output_width_ * output_height_; e++) {
          target_map[e] = 0.0;
        }
    }
    sk_id++;
  }*/

  bp_deltax_buffer.hint_ignore_content_ = true;
  sms2_bp_buffer.hint_ignore_content_ = true;
  weights_->delta.hint_ignore_content_ = true;
  bias_->delta.hint_ignore_content_ = true;
  input_->delta.hint_ignore_content_ = true;
  
  TensorMath::SMS(output_->delta, sms2_bp_buffer);
  
  for(unsigned int g = 0; g < group_; g++) {
    /*
    * 1. Backpropagation
    */
    if (backprop_enabled_)
      TensorMath::GEMM (true, true, false,
            (kernel_width_ * kernel_height_ * input_maps_) / group_,
            output_width_ * output_height_ * input_->data.samples(),
            output_maps_ / group_,
            1.0, weights_->data, (g * output_maps_) / group_, (kernel_width_ * kernel_height_ * input_maps_) / group_,
            sms2_bp_buffer, (g * output_maps_) / group_, output_width_ * output_height_ * input_->data.samples(),
            0.0, bp_deltax_buffer, (kernel_width_ * kernel_height_ * input_maps_ * g) / group_, output_width_ * output_height_ * input_->data.samples());
    
    /*
    * 2. Weight gradient calculation
    */
    if(local_lr_ > 0) {
      TensorMath::GEMM(true, false, true, output_maps_ / group_,
                       (kernel_width_ * kernel_height_ * input_maps_) / group_,
                       output_width_ * output_height_ * input_->data.samples(),
                       1.0, sms2_bp_buffer, (g * output_maps_) / group_,
                       output_width_ * output_height_ * input_->data.samples(),
                       im2col_ff_buffer, (kernel_width_ * kernel_height_ * input_maps_ * g) / group_,
                       output_width_ * output_height_ * input_->data.samples(),
                       0.0, weights_->delta, (g * output_maps_) / group_,
                       (kernel_width_ * kernel_height_ * input_maps_) / group_);
    } else {
      weights_->delta.Clear(0);
    }
  }
  /*
  * 3. Bias gradient calculation
  */
  if(local_lr_ > 0) {
    TensorMath::GEMV(true, false, output_maps_, output_width_ * output_height_ * input_->data.samples(), 1.0,
                     sms2_bp_buffer, 0, output_width_ * output_height_ * input_->data.samples(),
                     ones_, 0, 1, 0.0, bias_->delta, 0, 1);
  } else {
    bias_->delta.Clear(0);
  }

  
  if(backprop_enabled_)
    TensorMath::COL2IM(input_->delta, input_width_, input_height_, input_maps_, input_->data.samples(),
        kernel_width_, kernel_height_, stride_width_, stride_height_, pad_width_, pad_height_, bp_deltax_buffer);
}


void ConvolutionLayer::OnLayerConnect (const std::vector<Layer*> next_layers, bool no_init) {
	unsigned int next_layer_gain = 0;
	for (Layer* next_layer: next_layers)
		next_layer_gain += next_layer->Gain();

  unsigned int this_layer_gain = Gain();

  const datum range = sqrt (6) / sqrt (next_layer_gain + this_layer_gain);

#ifdef BUILD_OPENCL
  weights_->data.MoveToCPU();
#endif
  if (!no_init) {
    std::uniform_real_distribution<datum> dist_weights(-range, range);

    for (std::size_t i = 0; i < weights_->data.elements(); i++) {
      weights_->data[i] = dist_weights(rand_);
    }

    LOGDEBUG << "Updating weights: " << this_layer_gain << " -> "
      << next_layer_gain;
  }
  else {
    LOGDEBUG << "Skipping initialization";
  }
}

bool ConvolutionLayer::IsGPUMemoryAware() {
#ifdef BUILD_OPENCL_CONV
  return true;
#else
  return false;
#endif
}

}

