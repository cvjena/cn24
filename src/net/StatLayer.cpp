/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <cmath>

#include "Log.h"
#include "CombinedTensor.h"
#include "Init.h"
#include "StatLayer.h"

namespace Conv {

StatLayer::StatLayer() {
  LOGDEBUG << "Instance created!";
}

bool StatLayer::CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                               std::vector< CombinedTensor* >& outputs) {
  // Validate input node count
  if (inputs.size() != 2) {
    LOGERROR << "Need exactly 2 inputs to calculate stat!";
    return false;
  }

  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];

  // Check for null pointers
  if (first == nullptr || second == nullptr) {
    LOGERROR << "Null pointer node supplied";
    return false;
  }

  if (first->data.samples() != second->data.samples()) {
    LOGERROR << "Inputs need the same number of samples!";
    return false;
  }

  if (first->data.elements() != second->data.elements()) {
    LOGERROR << "Inputs need the same number of elements!";
    return false;
  }

  // Needs no outputs
  return true;
}

bool StatLayer::Connect (const std::vector< CombinedTensor* >& inputs,
                         const std::vector< CombinedTensor* >& outputs) {
  // Needs exactly two inputs to calculate the stat
  if (inputs.size() != 2)
    return false;

  // Also, the two inputs have to have the same number of samples and elements!
  // We ignore the shape for now...
  CombinedTensor* first = inputs[0];
  CombinedTensor* second = inputs[1];
  bool valid = first != nullptr && second != nullptr &&
               first->data.samples() == second->data.samples() &&
               first->data.elements() == second->data.elements() &&
               outputs.size() == 0;

  if (valid) {
    first_ = first;
    second_ = second;
  }

  return valid;
}

void StatLayer::FeedForward() {
  // Nothing to do here, because this will run after ErrorLayer
}

void StatLayer::BackPropagate() {
  // Nothing to do here
}

/*
 * Layer implementations here
 */

datum AccuracyLayer::CalculateStat() {
  datum acc_sum = 0;
  for (std::size_t s = 0; s < first_->data.elements(); s++) {
    std::size_t first_max = first_->data.Maximum (s);
    std::size_t second_max = second_->data.Maximum (s);
    if (first_max == second_max)
      acc_sum += 1;
  }

  return acc_sum / (datum) first_->data.elements();

}

datum BinAccuracyLayer::CalculateStat() {
  //System::viewer->show(&first_->data);
  //System::viewer->show(&second_->data);
  datum acc_sum = 0;
  for (std::size_t s = 0; s < first_->data.elements(); s++) {
    std::size_t first_sign = first_->data(s) > 0;
    std::size_t second_sign = second_->data(s) > 0;
    if (first_sign == second_sign)
      acc_sum += 1;
  }

  return acc_sum / (datum) first_->data.elements();
}

datum ErrorRateLayer::CalculateStat() {
  datum err_sum = 0;
  for (std::size_t s = 0; s < first_->data.elements(); s++) {
    std::size_t first_max = first_->data.Maximum (s);
    std::size_t second_max = second_->data.Maximum (s);
    if (first_max != second_max)
      err_sum += 1;
  }

  return err_sum / (datum) first_->data.elements();
}

datum BinErrorRateLayer::CalculateStat() {
  datum err_sum = 0;
  for (std::size_t s = 0; s < first_->data.elements(); s++) {
    std::size_t first_sign = first_->data(s) > 0;
    std::size_t second_sign = second_->data(s) > 0;
    if (first_sign != second_sign)
      err_sum += 1;
  }

  return err_sum / (datum) first_->data.elements();
}

}
