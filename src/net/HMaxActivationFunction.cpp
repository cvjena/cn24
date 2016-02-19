/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cmath>

#include "CombinedTensor.h"

#include "HMaxActivationFunction.h"

namespace Conv {
bool HMaxActivationFunction::CreateOutputs (
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
  
  // Create output
  CombinedTensor* output = new CombinedTensor (input->data.samples(),
                                               input->data.width(),
                                               input->data.height(),
                                               input->data.maps());
  // Tell network about the output
  outputs.push_back (output);
  
  return true;
}

bool HMaxActivationFunction::Connect (const CombinedTensor* input,
                                 CombinedTensor* output) {
  // Check dimensions for equality
  bool valid = input->data.samples() == output->data.samples() &&
  input->data.width() == output->data.width() &&
  input->data.height() == output->data.height() &&
  input->data.maps() == output->data.maps();
  
  if(!valid)
    return false;
  
  weights_ = new CombinedTensor(1, 2, 1, 1);
  weights_->data.Clear(1.0);
  weights_->delta.Clear();
  
  parameters_.push_back(weights_);
  
  return true;
}
  
void HMaxActivationFunction::FeedForward() {
  const datum a = *(weights_->data.data_ptr(0));
  const datum b = *(weights_->data.data_ptr(1));
  
  LOGDEBUG << "a: " << a << ", b:" << b;
  
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const() [element];

    // Calculate sigmoid function
    const datum output_data = 1.0 / (1.0 + exp(-(a * input_data + b)));
    output_->data.data_ptr() [element] = output_data;
  }
}
  
void HMaxActivationFunction::BackPropagate() {
  // Calculate gradient w.r.t. input
  const datum a = weights_->data.data_ptr_const()[0];
  const datum b = weights_->data.data_ptr_const()[1];
  
  // weights the regularization term
  const datum lambda_sparse_regularization = 0.1;

  // TODO: This layer needs to produce a loss (aka regularizer) as well
  // that is added directly to the objective

  // calculate part of the derivative of the KL criterion
  // which is constant for all dimensions
  // https://figss.uni-frankfurt.de/~triesch/publications/Triesch-ICANN2005.pdf
  datum delta_a = -((datum)(input_->data.elements()))/a * lambda_sparse_regularization;
  datum delta_b = -(datum)(input_->data.elements()) * lambda_sparse_regularization;
#pragma omp parallel for default(shared)
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const() [element];
    const datum output_data = output_->data.data_ptr_const() [element];
    const datum output_delta = output_->delta.data_ptr_const ()[element];
    const datum output_mix = (1-output_data) * output_data;

    // Calculate derivative
    // o = (1 + exp(-ax - b))^{-1}
    // https://en.wikipedia.org/wiki/Logistic_function
    // do/dx = o (1-o) * a
    // do/da = o (1-o) * x
    // do/db = o (1-o)
    datum input_delta = output_delta * a * output_mix; 
    
    // Calculate gradient of the output w.r.t. to weights
    delta_a += output_mix * input_data;
    delta_b += output_mix;

    // handle the KL divergence regularization;
    // We need to add the derivative of eq. (4) w.r.t. x here to input_delta
    // weighted by lambda_sparse_regularization
    // d/(dx) o log o = do/dx * log o + o * do/dx * o^-1
    //                = do/dx ( log o + 1 )
    input_delta += lambda_sparse_regularization * ( output_mix * (log(output_data) + 1) );
    input_delta += lambda_sparse_regularization * output_mix / mu_;

    input_->delta.data_ptr() [element] = input_delta;
    
    // Calculate gradient of KL-divergence w.r.t. weights
    delta_a += (-input_data + (2.0 + 1.0/mu_) * input_data * output_data - input_data * output_data * output_data / mu_) 
        * lambda_sparse_regularization;
    delta_b += ((2.0 + 1.0/mu_) * output_data - output_data * output_data / mu_) 
        * lambda_sparse_regularization;
  } 
  
  weights_->delta.data_ptr()[0] = local_lr_ * delta_a;
  weights_->delta.data_ptr()[1] = local_lr_ * delta_b;
  LOGDEBUG << "delta a: " << delta_a << ", delta b:" << delta_b;
}


}
