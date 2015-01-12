/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file ConfusionMatrixLayer.h
 * \class ConfusionMatrixLayer
 * \brief Represents a layer that calculates a confusion matrix
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_CONFUSIONMATRIXLAYER_H
#define CONV_CONFUSIONMATRIXLAYER_H

#include <string>
#include <vector>

#include "Layer.h"

namespace Conv {

class ConfusionMatrixLayer: public Layer {
public:
  explicit ConfusionMatrixLayer(std::vector<std::string>& names,
                                const unsigned int classes);
  ~ConfusionMatrixLayer();
  void Print (std::string prefix, bool training);
  void PrintCSV (std::ostream& output);
  void Reset();

  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs);
  void FeedForward();
  void BackPropagate();

  inline void SetDisabled (bool disabled = false) {
    disabled_ = disabled;
  }

private:
  unsigned int classes_;
  std::vector<std::string>& names_;
  bool disabled_ = false;
  
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
  CombinedTensor* third_ = nullptr;
  
  long double* matrix_ = nullptr;
  long double total_ = 0;
  long double right_ = 0;
  long double* per_class_ = nullptr;
};

}

#endif
