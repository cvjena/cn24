/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cmath>
#include "StatAggregator.h"
#include "SparsityLossLayer.h"

namespace Conv {

SparsityLossLayer::SparsityLossLayer(JSON configuration) : Layer(configuration){
  loss_weight_ = 1.0;
  if(configuration.count("weight") == 1 && configuration["weight"].is_number()) {
    loss_weight_ = configuration["weight"];
  }

  // Create and register sparsity stat
  stat_descriptor_.description = "L1/L2 Ratio";
  stat_descriptor_.nullable = true;
  stat_descriptor_.unit = "1";
  stat_descriptor_.init_function = [](Stat& stat) {stat.is_null = true; stat.value = 0.0;};
  stat_descriptor_.update_function = [](Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
  stat_descriptor_.output_function = [](HardcodedStats& hc_stats, Stat& stat) -> Stat {
    Stat return_stat; return_stat.is_null = true;
    if (hc_stats.iterations > 0) {
      double d_iterations = (double)hc_stats.iterations;
      return_stat.value = stat.value / d_iterations;
      return_stat.is_null = false;
    }
    return return_stat;
  };
  stat_id_ = System::stat_aggregator->RegisterStat(&stat_descriptor_);
}

bool SparsityLossLayer::CreateOutputs(const std::vector<CombinedTensor *> &inputs,
                                      std::vector<CombinedTensor *> &outputs) {
  UNREFERENCED_PARAMETER(outputs);
  if(inputs.size() != 1) {
    LOGERROR << "Needs exacly one input!";
    return false;
  }

  return true;
}

bool SparsityLossLayer::Connect(const std::vector<CombinedTensor *> &inputs,
                                const std::vector<CombinedTensor *> &outputs, const NetStatus *net) {
  UNREFERENCED_PARAMETER(outputs);
  UNREFERENCED_PARAMETER(net);
  if(inputs.size() != 1) {
    LOGERROR << "Needs exacly one input!";
    return false;
  }
  first_ = inputs[0];

  return true;
}

void SparsityLossLayer::FeedForward() {
  for (unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    // 1. Calculate sums
    datum sum_of_squares = 0;
    datum sum_of_abs = 0;

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

    const datum inv_l2_norm = (sum_of_squares > zero_threshold) ? 1.0/std::sqrt(sum_of_squares) : 0.0;
    const datum squares_32 = std::pow(sum_of_squares, 1.5);

    const datum abs_32_fraction = sum_of_abs / squares_32;

    // 2. Calculate derivatives
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
          *first_->delta.data_ptr(x, y, map, sample) = loss_weight_ * delta;
        }
      }
    }

    System::stat_aggregator->Update(stat_id_, (sum_of_abs * inv_l2_norm) / (datum)first_->data.samples());
  }
}

void SparsityLossLayer::BackPropagate() {

}

datum SparsityLossLayer::CalculateLossFunction() {
  datum loss_sum = 0;

  for (unsigned int sample = 0; sample < first_->data.samples(); sample++ ) {
    // 1. Calculate sums
    datum sum_of_squares = 0;
    datum sum_of_abs = 0;

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
    const datum inv_l2_norm = (sum_of_squares > zero_threshold) ? 1.0/std::sqrt(sum_of_squares) : 0.0;
    loss_sum += sum_of_abs * inv_l2_norm;
  }
  return loss_weight_ * loss_sum;
}

}
