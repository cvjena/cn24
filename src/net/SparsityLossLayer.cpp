/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cmath>
#include "SparsityLossLayer.h"

namespace Conv {

SparsityLossLayer::SparsityLossLayer(JSON configuration) : Layer(configuration){
  loss_weight_ = 1.0;
  if(configuration.count("weight") == 1 && configuration["weight"].is_number()) {
    loss_weight_ = configuration["weight"];
  }
}

bool SparsityLossLayer::CreateOutputs(const std::vector<CombinedTensor *> &inputs,
                                      std::vector<CombinedTensor *> &outputs) {
  if(inputs.size() != 1) {
    LOGERROR << "Needs exacly one input!";
    return false;
  }

  return true;
}

bool SparsityLossLayer::Connect(const std::vector<CombinedTensor *> &inputs,
                                const std::vector<CombinedTensor *> &outputs, const NetStatus *net) {
  if(inputs.size() != 1) {
    LOGERROR << "Needs exacly one input!";
    return false;
  }
  first_ = inputs[0];

  return true;
}

void SparsityLossLayer::FeedForward() {

}

void SparsityLossLayer::BackPropagate() {
  // 1. Calculate sums
  datum sum_of_squares = 0;
  datum sum_of_abs = 0;

  for (unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    for (unsigned int map = 0; map < first_->data.maps(); map++) {
      for (unsigned int y = 0; y < first_->data.height(); y++) {
        for (unsigned int x = 0; x < first_->data.width(); x++) {
          const datum first =
            *first_->data.data_ptr_const(x, y, map, sample);

          sum_of_squares += first * first;
          sum_of_abs += std::abs(first);
        }
      }
    }
  }

  const datum inv_l2_norm = (sum_of_squares > zero_threshold) ? 1.0/std::sqrt(sum_of_squares) : 0.0;
  const datum squares_32 = std::pow(sum_of_squares, 1.5);

  const datum abs_32_fraction = sum_of_abs / squares_32;

  // 2. Calculate derivatives
  for (unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    for (unsigned int map = 0; map < first_->data.maps(); map++) {
      for (unsigned int y = 0; y < first_->data.height(); y++) {
        for (unsigned int x = 0; x < first_->data.width(); x++) {
          const datum first =
              *first_->data.data_ptr_const(x, y, map, sample);
          datum delta = - first * abs_32_fraction;
          if(first > zero_threshold) {
            delta += inv_l2_norm;
          } else if (first < -zero_threshold) {
            delta -= inv_l2_norm;
          }
          *first_->delta.data_ptr(x, y, map, sample) = delta;
        }
      }
    }
  }
}

datum SparsityLossLayer::CalculateLossFunction() {
  // 1. Calculate sums
  datum sum_of_squares = 0;
  datum sum_of_abs = 0;

  for (unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    for (unsigned int map = 0; map < first_->data.maps(); map++) {
      for (unsigned int y = 0; y < first_->data.height(); y++) {
        for (unsigned int x = 0; x < first_->data.width(); x++) {
          const datum first =
            *first_->data.data_ptr_const(x, y, map, sample);

          sum_of_squares += first * first;
          sum_of_abs += std::abs(first);
        }
      }
    }
  }

  return sum_of_abs / std::sqrt(sum_of_squares);
}

}
