/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file SupportsDropoutLayer.h
 * \class SupportsDropoutLayer
 * \brief Interface for layers that support weight changing for dropout purposes.
 * 
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_SUPPORTSDROPOUTLAYER_H
#define CONV_SUPPORTSDROPOUTLAYER_H

#include "Config.h"
#include "Log.h"

namespace Conv {
  
class SupportsDropoutLayer {
public:
  void SetWeightFactor(const datum factor) {
    LOGDEBUG << "Setting weight factor to " << factor;
    weight_factor_ = factor;
  }
protected:
  datum weight_factor_ = 1.0;
};
}

#endif