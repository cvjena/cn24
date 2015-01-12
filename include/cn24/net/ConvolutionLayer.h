/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file ConvolutionLayer.h
 * \class ConvolutionLayer
 * \brief Represents a layer that learns convolution kernels.
 * 
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_CONVOLUTIONLAYER_H
#define CONV_CONVOLUTIONLAYER_H

#include <random>

#include "Layer.h"
#include "SimpleLayer.h"
#include "SupportsDropoutLayer.h"

#ifdef BUILD_OPENCL
#define BUILD_OPENCL_CONV
#endif

namespace Conv {

class ConvolutionLayer : public SimpleLayer, public SupportsDropoutLayer {
public:
  /**
   * \brief Constructs a ConvolutionLayer.
   * 
   * \param kwidth Width of the kernels
   * \param kheight Height of the kernels
   * \param output_maps Number of output feature maps
   * \param seed Random seed for weight generation
   */
  ConvolutionLayer(const unsigned int kwidth, const unsigned int kheight,
                   const unsigned int output_maps, const int seed = 0);
  
  // Implementations for SimpleLayer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const CombinedTensor* input, CombinedTensor* output);
  void FeedForward();
  void BackPropagate();
  
  void OnLayerConnect (Layer* next_layer);
  
  inline unsigned int Gain() {
    return kernel_width_ * kernel_height_ * input_maps_;
  }
  
#ifdef BUILD_OPENCL_CONV
  bool IsOpenCLAware() { return true; }
#endif
private:
  void im2colff();
  void col2imff();
  
  void im2colbp();
  void col2imbp();
  
  Tensor im2col_ff_buffer;
  Tensor ff_output_buffer;
  Tensor bp_deltax_buffer;
  Tensor bp_deltay_buffer;
  
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
#ifdef BUILD_OPENCL_CONV
  Tensor delta_buffer_;
  Tensor bias_buffer_;
#endif
  
  std::mt19937 rand_;
};

}

#endif