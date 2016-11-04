/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "AdamOptimizer.h"

#include <cmath>

namespace Conv {
AdamOptimizer::AdamOptimizer(JSON descriptor) : Optimizer(descriptor) {
  JSON_TRY_DATUM(step_size_, descriptor, "ad_step_size", 0.001);
  JSON_TRY_DATUM(beta1_, descriptor, "ad_beta1", 0.9);
  JSON_TRY_DATUM(beta2_, descriptor, "ad_beta2", 0.999);
  JSON_TRY_DATUM(epsilon_, descriptor, "ad_epsilon", 0.00000001);
}

void AdamOptimizer::Step(std::vector<Tensor> &buffers, CombinedTensor* parameters,
                        unsigned int reported_iteration) {
  UNREFERENCED_PARAMETER(reported_iteration);
  // Do not use the reported iteration, it doesn't reset!
  const float actual_iteration = iterations_since_reset_ + 1;
  Tensor& first_moment = buffers[0];
  Tensor& second_moment = buffers[1];

  const datum current_step_size = step_size_; /// sqrtf(actual_iteration);

  for(std::size_t p = 0; p < parameters->data.elements(); p++) {
    const datum g_t = parameters->delta(p);
    const datum m_t = beta1_ * first_moment(p) + ((Conv::datum)1.0 - beta1_) * g_t;
    const datum v_t = beta2_ * second_moment(p) + ((Conv::datum)1.0 - beta2_) * g_t * g_t;

    const datum m_hat_t = m_t / ((Conv::datum)1.0 - (Conv::datum)pow(beta1_, actual_iteration));
    const datum v_hat_t = v_t / ((Conv::datum)1.0 - (Conv::datum)pow(beta2_, actual_iteration));

    if(actual_iteration > 2) {
      parameters->data[p] = parameters->data(p) - current_step_size * (m_hat_t / (sqrtf(v_hat_t) + epsilon_));
    } else {
      parameters->data[p] = parameters->data(p) - current_step_size * g_t;
    }
    // Save moments
    first_moment[p] = m_t;
    second_moment[p] = v_t;
  }
}
}