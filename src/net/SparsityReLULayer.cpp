/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cmath>

#include "CombinedTensor.h"
#include "ConfigParsing.h"

#include "SparsityReLULayer.h"

namespace Conv {
  
SparsityReLULayer::SparsityReLULayer(const datum lambda, const datum kl_loss_weight)
  : SimpleLayer(JSON::object()), lambda_(lambda), kl_loss_weight_(kl_loss_weight) {

}
  
SparsityReLULayer::SparsityReLULayer(JSON configuration)
  : SimpleLayer(configuration) {
  alpha_ = 0.5;
  lambda_ = 1.0;
  kl_loss_weight_ = 0.0;
  other_loss_weight_ = 0.0;

	if(configuration.count("lambda") == 1 && configuration["lambda"].is_number()) {
		lambda_ = configuration["lambda"];
	}

  if(configuration.count("alpha") == 1 && configuration["alpha"].is_number()) {
    alpha_ = configuration["alpha"];
  }

  if(configuration.count("kl_weight") == 1 && configuration["kl_weight"].is_number()) {
		kl_loss_weight_ = configuration["kl_weight"];
	}

  if(configuration.count("other_weight") == 1 && configuration["other_weight"].is_number()) {
    other_loss_weight_ = configuration["other_weight"];
  }


  if(configuration.count("llr") == 1 && configuration["llr"].is_number()) {
    local_lr_ = configuration["llr"];
  } else if(kl_loss_weight_ == 0.0 && other_loss_weight_ == 0.0) {
    local_lr_ = 0.0;
  }

  SetLocalLearningRate(local_lr_);

}
  
bool SparsityReLULayer::CreateOutputs (
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

bool SparsityReLULayer::Connect (const CombinedTensor* input,
                                 CombinedTensor* output) {
  // Check dimensions for equality
  bool valid = input->data.samples() == output->data.samples() &&
  input->data.width() == output->data.width() &&
  input->data.height() == output->data.height() &&
  input->data.maps() == output->data.maps();
  
  if(!valid)
    return false;
  
  weights_ = new CombinedTensor(1, 2, 1, 1);
  weights_->data.Clear();
  weights_->delta.Clear();

  weights_->data.data_ptr()[0] = 1.0;
  weights_->data.data_ptr()[1] = 0.0;

  parameters_.push_back(weights_);
  
  return true;
}
  
void SparsityReLULayer::FeedForward() {
  const datum a = *(weights_->data.data_ptr(0));
  const datum b = *(weights_->data.data_ptr(1));
  
  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const() [element];
    const datum transformed_input = a * input_data + b;

    if(transformed_input >= 0) {
      output_->data.data_ptr() [element] = transformed_input + alpha_;
    } else {
      output_->data.data_ptr() [element] = alpha_ * exp(transformed_input  / alpha_);
    }
  }
}
  
void SparsityReLULayer::BackPropagate() {
  const datum a = weights_->data.data_ptr_const()[0];
  const datum b = weights_->data.data_ptr_const()[1];

  datum a_delta_kl = 0.0;
  datum b_delta_kl = 0.0;
  datum a_delta_other = 0.0;
  datum b_delta_other = 0.0;

  const datum inv_elements_per_sample = 1.0 / (datum)(input_->data.elements() / input_->data.samples());

  for (std::size_t element = 0; element < input_->data.elements(); element++) {
    const datum input_data = input_->data.data_ptr_const() [element];
    const datum output_delta = output_->delta.data_ptr_const ()[element];
    const datum transformed_input = a * input_data + b;

    datum input_delta = a;

    a_delta_kl -= 1.0 / a;

    if (transformed_input >= 0) {
      // Gradient of KL-divergence wrt a
      a_delta_kl += lambda_ * input_data;
      // Gradient of KL-divergence wrt b
      b_delta_kl += lambda_;

      // Gradient of output wrt a
      a_delta_other += input_data;
      // Gradient of output wrt b
      b_delta_other += 1;
    } else {
      // Gradient of output wrt input
      input_delta *= exp(transformed_input / alpha_);

      // Gradient of KL-divergence wrt a
      a_delta_kl -= (input_data / alpha_) - lambda_ * input_data * exp(transformed_input / alpha_);
      // Gradient of KL-divergence wrt b
      b_delta_kl += (-1.0 / alpha_) + lambda_ * exp(transformed_input / alpha_);

      // Gradient of output wrt a
      a_delta_other += input_data * exp(transformed_input / alpha_);
      // Gradient of output wrt b
      b_delta_other += exp(transformed_input / alpha_);
    }

    // Save gradient wrt input
    input_->delta.data_ptr() [element] = output_delta * input_delta;
  }

  // Save gradient wrt a and b
  weights_->delta.data_ptr()[0] = inv_elements_per_sample * kl_loss_weight_ * a_delta_kl + other_loss_weight_ * a_delta_other;
  weights_->delta.data_ptr()[1] = inv_elements_per_sample * kl_loss_weight_ * b_delta_kl + other_loss_weight_ * b_delta_other;
}
  
datum SparsityReLULayer::CalculateLossFunction() {
  return 0;
}


}
