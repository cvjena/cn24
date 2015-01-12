/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file DropoutLayer.h
 * \class DropoutLayer
 * \brief This layer implements the 'dropout' regularization method
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_DROPOUTLAYER_H
#define CONV_DROPOUTLAYER_H

#include <random>
#include "NonLinearityLayer.h"
#include "SupportsDropoutLayer.h"

namespace Conv {

class DropoutLayer: public NonLinearityLayer {
public:
  DropoutLayer (const datum dropout_frac,
                SupportsDropoutLayer* supports_dropout_layer,
                const int seed = 0) :
    dropout_frac_ (dropout_frac), supports_dropout_layer_(supports_dropout_layer),
    generator_ (seed), dist_ (0.0, 1.0) {
    LOGDEBUG << "Instance created";
    if (seed == 0) {
      LOGWARN << "Random seed is zero";
    }
  }
  void FeedForward();
  void BackPropagate();
  inline void SetDropoutEnabled (const bool do_dropout) {
    do_dropout_ = do_dropout;
    if(do_dropout) {
      LOGDEBUG << "Dropout enabled";
      if(supports_dropout_layer_ != nullptr)
        supports_dropout_layer_->SetWeightFactor(1.0);
    } else {
      LOGDEBUG << "Dropout disabled";
      if(supports_dropout_layer_ != nullptr)
        supports_dropout_layer_->SetWeightFactor(1.0-dropout_frac_);
    }
  }
private:
  // Settings
  datum dropout_frac_;
  SupportsDropoutLayer* supports_dropout_layer_ = nullptr;

  // State
  std::mt19937 generator_;
  std::uniform_real_distribution<datum> dist_;
  bool do_dropout_ = false;
};

}

#endif
