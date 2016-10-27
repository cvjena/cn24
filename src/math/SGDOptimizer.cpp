/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "SGDOptimizer.h"

#include <sstream>

namespace Conv {
SGDOptimizer::SGDOptimizer(JSON descriptor) : Optimizer(descriptor) {
  JSON_TRY_DATUM(learning_rate_, descriptor, "learning_rate", 0.0001);
  JSON_TRY_DATUM(momentum_, descriptor, "gd_momentum", 0.9);
  JSON_TRY_DATUM(learning_rate_exponent_, descriptor, "learning_rate_exponent", 0.75);
  JSON_TRY_DATUM(learning_rate_gamma_, descriptor, "learning_rate_gamma", 0.0003);
}

void SGDOptimizer::Step(std::vector<Tensor> &buffers, CombinedTensor* parameters,
                        unsigned int iteration) {
  const datum current_learning_rate =
      learning_rate_ * (datum)pow(1.0 + learning_rate_gamma_ * (datum)iteration, learning_rate_exponent_);

  Tensor& momentum_buffer = buffers[0];
  for(std::size_t p = 0; p < parameters->data.elements(); p++) {
    const datum last_step = momentum_buffer(p);
    const datum this_step = current_learning_rate * parameters->delta(p) + momentum_ * last_step;
    parameters->data[p] -= this_step;
    momentum_buffer[p] = this_step;
  }
}

std::string SGDOptimizer::GetStatusDescription(unsigned int iteration) {
  const datum current_learning_rate =
      learning_rate_ * (datum)pow(1.0 + learning_rate_gamma_ * (datum)iteration, learning_rate_exponent_);
  std::stringstream ss;
  ss << "it: " << iteration << ", lr0: " << learning_rate_ << ", lr_exp: " << learning_rate_exponent_;
  ss << ", lr_gamma: " << learning_rate_gamma_ << ", mom: " << momentum_ << ", lr_current: " << current_learning_rate;
  return ss.str();
}
}