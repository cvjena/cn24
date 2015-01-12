/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file StatLayer.h
 * \class StatLayer
 * \brief Layer that calculates a statistical measure on the output
 * 
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_STATLAYER_H
#define CONV_STATLAYER_H

#include <string>

#include "Layer.h"

namespace Conv {
  
#define STAT_LAYER(name) class name##Layer : public StatLayer {\
public:\
  datum CalculateStat();\
  std::string stat_name() { return std::string(#name); }\
};

class StatLayer: public Layer {
public:
  StatLayer();
  /*
   * \brief Calculate the statistic
   */
  virtual datum CalculateStat() = 0;
  virtual std::string stat_name() = 0;
  
  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs);
  void FeedForward();
  void BackPropagate(); 
  
protected:
  CombinedTensor* first_ = nullptr;
  CombinedTensor* second_ = nullptr;
};

STAT_LAYER(BinAccuracy)
STAT_LAYER(BinErrorRate)
STAT_LAYER(Accuracy)
STAT_LAYER(ErrorRate)

}

#endif