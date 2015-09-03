/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file LossFunctionLayer.h
 * @class LossFunctionLayer
 * @brief This interface connects the Trainer with an abstract loss function.
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_LOSSFUNCTIONLAYER_H
#define CONV_LOSSFUNCTIONLAYER_H

#include "../util/Config.h"

namespace Conv {

class LossFunctionLayer {
public:
  /**
   * @brief Calculate the loss function after a complete forward pass.
   */
  virtual datum CalculateLossFunction() = 0;
};

}

#endif
