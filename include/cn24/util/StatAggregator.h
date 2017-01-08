/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file StatAggregator.h
 * @brief Collects data from various sources and aggregates them into a statistic
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_STATAGGREGATOR_H
#define CONV_STATAGGREGATOR_H

#include <functional>
#include <vector>
#include <string>
#include <chrono>
#include <climits>

#include "Config.h"

namespace Conv
{
// Forward declarations
class StatSink;
class Trainer;
class NetStatus;

// Hardcoded stats
struct HardcodedStats {
  double seconds_elapsed = 0.0;
  unsigned long iterations = 0UL;
  unsigned long weights = 0UL;
  unsigned long epoch = 0UL;
  bool is_training = false;
  std::string current_experiment = "unnamed";
  std::string current_testing_dataset = "unnamed";
  
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
  std::string description = "";
  std::string unit = "";
  
  // Lambdas for processing
  std::function<void(Stat&)> init_function = [] (Stat& stat) {UNREFERENCED_PARAMETER(stat);};
  std::function<void(Stat&,double)> update_function = [] (Stat& stat, double user_value) {UNREFERENCED_PARAMETER(stat); UNREFERENCED_PARAMETER(user_value);};
  std::function<Stat(HardcodedStats&, Stat&)> output_function =
    [] (HardcodedStats& hc_stats, Stat& stat) -> Stat {UNREFERENCED_PARAMETER(hc_stats); return stat;};
    
  // For easy access
  unsigned int stat_id = UINT_MAX;
};

class StatAggregator {
  friend class Trainer;
  friend class NetStatus;
public:
  unsigned int RegisterStat(StatDescriptor* stat_descriptor);
  unsigned int RegisterSink(StatSink* stat_sink);
  void Initialize();
  
  void Update(unsigned int stat_id, double user_value);
  void Generate();
  
  void StartRecording();
  void StopRecording();
  void Reset();

  void Snapshot();
  
  void SetCurrentExperiment(std::string current_experiment);
  void SetCurrentTestingDataset(unsigned int current_testing_dataset);
  HardcodedStats hardcoded_stats_;
private:
  // State
  enum StatAggregatorState {
    STOPPED, RECORDING, INIT } state_ = INIT;
  std::chrono::time_point<std::chrono::system_clock> start_time_;
  
  // Stats
  std::vector<Stat> stats_;
  
  // Descriptors
  std::vector<StatDescriptor*> stat_descriptors_;
  unsigned int stat_descriptor_count_ = 0;
  
  // Sinks
  std::vector<StatSink*> stat_sinks_;
  unsigned int stat_sink_count_ = 0;
};  

}

#endif
