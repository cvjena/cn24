/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file CSVStatSink.h
 * @brief Gets data from StatAggregator and processes it into a CSV file
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CSVSTATSINK_H
#define CONV_CSVSTATSINK_H

#include <functional>
#include <vector>
#include <iomanip>
#include <fstream>

#include "Config.h"
#include "Log.h"

#include "StatAggregator.h"
#include "StatSink.h"

namespace Conv
{
class CSVStatSink : public StatSink {
public:
  ~CSVStatSink() { if(csv_stream_ != nullptr) {csv_stream_->close(); delete csv_stream_; }}
  virtual void Initialize(std::vector<StatDescriptor*>& stat_descriptors);
  virtual void Process(HardcodedStats& hardcoded_stats, std::vector<Stat*>& stats);
  virtual void SetCurrentExperiment(std::string current_experiment);
  virtual void SetCurrentTestingDataset(unsigned int current_dataset);

private:
  std::vector<StatDescriptor*> stat_descriptors_;
  std::ofstream* csv_stream_ = nullptr;
  unsigned int current_dataset_;
};

}

#endif