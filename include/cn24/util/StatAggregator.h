/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file StatAggregator.h
 * @brief Collects data from varios sources and aggregates them into a statistic
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_STATAGGREGATOR_H
#define CONV_STATAGGREGATOR_H

#include <functional>
#include <vector>
#include <string>
#include <chrono>

#include "Config.h"

namespace Conv
{
// Forward declaration
class StatSink;

// Hardcoded stats
struct HardcodedStats {
  double seconds_elapsed = 0.0;
  unsigned long iterations = 0UL;
  unsigned long weights = 0UL;
  
  void Reset() {
    seconds_elapsed = 0.0;
    iterations = 0UL;
    weights = 0UL;
  }
};

struct Stat {
  double value = 0.0;
  bool is_null = false;
};

struct StatDescriptor {
  bool nullable = false;
  
  // Lambdas for processing
  std::function<void(Stat&)> init_function = [] (Stat& stat) {};
  std::function<void(Stat&)> update_function = [] (Stat& stat) {};
  std::function<Stat(Stat&, HardcodedStats&)> output_function =
    [] (Stat& stat, HardcodedStats& hc_stats) -> Stat {return stat;};
};

class StatAggregator {
public:
  void Update(unsigned int stat_id);
  
  void StartRecording();
  void StopRecording();
  void Reset();
  
private:
  // State
  bool is_recording_ = false;
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  
  // Stats
  HardcodedStats hardcoded_stats_;
  std::vector<Stat> stats_;
  
  // Descriptors
  std::vector<StatDescriptor*> stat_descriptors_;
  unsigned int stat_descriptor_count_ = 0;
};  

}

#endif