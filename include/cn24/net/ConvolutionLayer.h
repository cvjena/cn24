/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file ConvolutionLayer.h
 * @class ConvolutionLayer
 * @brief Represents a layer that learns convolution kernels.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CONVOLUTIONLAYER_H
#define CONV_CONVOLUTIONLAYER_H

#include <random>
#include <sstream>
#include <string>

#include "Layer.h"
#include "SimpleLayer.h"

namespace Conv {

class ConvolutionLayer : public SimpleLayer {
public:
  /**
   * @brief Constructs a ConvolutionLayer.
   * 
   * @param kwidth Width of the kernels
   * @param kheight Height of the kernels
   * @param output_maps Number of output feature maps
   * @param stride Stride of the convolution
   * @param seed Random seed for weight generation
   * @param dropout_fraction Propability of a feature map to be dropped out
   */
  ConvolutionLayer(const unsigned int kwidth, const unsigned int kheight,
                   const unsigned int output_maps, const unsigned int stride_width_ = 1,
                   const unsigned int stride_height = 1, const unsigned int pad_width = 0,
                   const unsigned int pad_height = 0, const unsigned int group = 1,
                   const int seed = 0, const datum dropout_fraction = 0.0 );
  
  explicit ConvolutionLayer(JSON configuration);
  
  ~ConvolutionLayer() {
    if(weights_ != nullptr)
      delete weights_;
    if(bias_ != nullptr)
      delete bias_;
  }
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
  
  void OnLayerConnect (const std::vector<Layer*> next_layer, bool no_init);
  
  inline unsigned int Gain() {
    return kernel_width_ * kernel_height_ * input_maps_;
  }

	inline std::string GetLayerDescription() {
		std::ostringstream ss;
		ss << "Convolutional Layer (" << output_maps_ << " kernels @ " << kernel_width_ << "x" << kernel_height_ << ")";
		return ss.str();
	}
  
  bool IsGPUMemoryAware();
private:
  Tensor im2col_ff_buffer;
  Tensor sms_ff_buffer;
  Tensor sms2_bp_buffer;
  Tensor bp_deltax_buffer;
  
  Tensor ones_;
  
  unsigned int input_maps_ = 0;
  unsigned int output_maps_ = 0;
  
  unsigned int kernel_width_ = 0;
  unsigned int kernel_height_ = 0;
  
  unsigned int input_width_ = 0;
  unsigned int input_height_ = 0;
  
  unsigned int output_width_ = 0;
  unsigned int output_height_ = 0;
  
  CombinedTensor* weights_ = nullptr;
  CombinedTensor* bias_ = nullptr;
  Tensor delta_buffer_;
  Tensor bias_buffer_;
  Tensor dropout_mask_;
  
  std::mt19937 rand_;

  unsigned int stride_width_ = 0;
  unsigned int stride_height_ = 0;
  unsigned int pad_width_ = 0;
  unsigned int pad_height_ = 0;
  unsigned int group_ = 0;
  datum dropout_fraction_ = 0.0;
};

}

#endif
