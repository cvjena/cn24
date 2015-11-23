/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file ConsoleStatSink.h
 * @brief Gets data from StatAggregator and processes it
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CONSOLESTATSINK_H
#define CONV_CONSOLESTATSINK_H

#include <functional>
#include <vector>
#include <iomanip>

#include "Config.h"
#include "Log.h"

#include "StatAggregator.h"
#include "StatSink.h"

namespace Conv
{
// Forward declaration
class ConsoleStatSink : public StatSink {
public:
  virtual void Initialize(std::vector<StatDescriptor*>& stat_descriptors) {
    stat_descriptors_ = stat_descriptors;
    LOGDEBUG << "Initializing ConsoleStatSink. Registered Stats:";
    for(unsigned int s = 0; s < stat_descriptors_.size(); s++) {
      LOGDEBUG << " - " << stat_descriptors_[s]->description;
    }
  }
  virtual void Process(HardcodedStats& hardcoded_stats, std::vector<Stat*>& stats) {
    LOGDEBUG << "Stats for epoch " << hardcoded_stats.epoch << ":";
    for(unsigned int s = 0; s < stat_descriptors_.size(); s++) {
      LOGDEBUG << std::setw(24) << stat_descriptors_[s]->description << ": " << std::setw(24) << stats[s]->value << " " << stat_descriptors_[s]->unit;
    }
  }
  
private:
  std::vector<StatDescriptor*> stat_descriptors_;
};

}

#endif