/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file StatLayer.h
 * @class StatLayer
 * @brief Layer that calculates a statistical measure on the output
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_STATLAYER_H
#define CONV_STATLAYER_H

#include <string>

#include "Layer.h"

namespace Conv {
  
class StatLayer {
public:
  virtual void UpdateAll() = 0;
  virtual void Print(std::string prefix, bool training) = 0;
  virtual void Reset() = 0;
	virtual void SetDisabled(bool disabled) = 0;
};

}

#endif
