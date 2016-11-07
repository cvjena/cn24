/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file StatSink.h
 * @brief Gets data from StatAggregator and processes it
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_STATSINK_H
#define CONV_STATSINK_H

#include <functional>
#include <vector>

#include "Config.h"

#include "StatAggregator.h"

namespace Conv
{
// Forward declaration
class StatSink {
public:
  virtual void Initialize(std::vector<StatDescriptor*>& stat_descriptors) = 0;
  virtual void SetCurrentExperiment(std::string current_experiment) = 0;
  virtual void SetCurrentTestingDataset(unsigned int current_dataset) = 0;
  virtual void Process(HardcodedStats& hardcoded_stats, std::vector<Stat*>& stats) = 0;
};

}

#endif