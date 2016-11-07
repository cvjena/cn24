/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "StatAggregator.h"
#include "StatSink.h"

#include <chrono>

namespace Conv {

unsigned int StatAggregator::RegisterSink(StatSink* stat_sink)
{
  stat_sinks_.push_back(stat_sink);
  return stat_sink_count_++;
}

unsigned int StatAggregator::RegisterStat(StatDescriptor* stat_descriptor)
{
  stat_descriptors_.push_back(stat_descriptor);
  stat_descriptor->stat_id = stat_descriptor_count_;
  return stat_descriptor_count_++;
}

void StatAggregator::Initialize()
{
  if(state_!=INIT)
    return;
  
  // Initialize all StatSinks
  for(unsigned int s = 0; s < stat_sink_count_; s++) {
    stat_sinks_[s]->Initialize(stat_descriptors_);
  }
  
  // Initialize all statistics
  for(unsigned int s = 0; s < stat_descriptor_count_; s++) {
    Stat stat;
    stats_.push_back(stat);
  }
  state_ = STOPPED;
  
  // Reset statistics
  Reset();
  
  // Send SetCurrentExperiment "message" to all StatSinks at least once before processing
  SetCurrentExperiment(hardcoded_stats_.current_experiment);
}

void StatAggregator::Generate()
{
  std::vector<Stat*> output_stats;
  
  for(unsigned int s = 0; s < stat_descriptor_count_; s++) {
    // We will not check for output_function's validity. We need its output.
    Stat* output_stat = new Stat;
    *output_stat = stat_descriptors_[s]->output_function(hardcoded_stats_, stats_[s]);
    output_stats.push_back(output_stat);
  }
  
  // Call all StatSinks' Process method
  for(unsigned int s = 0; s < stat_sink_count_; s++) {
    stat_sinks_[s]->Process(hardcoded_stats_, output_stats);
  }
  
  
  // Free all the allocated memory
  for(unsigned int s = 0; s < stat_descriptor_count_; s++) {
    delete (output_stats[s]);
  }
}


void StatAggregator::Update(unsigned int stat_id, double user_value)
{
  // Ignore this call if not recording
  if(state_ != RECORDING)
    return;
  
  if(stat_id < stat_descriptor_count_) {
    // We will not check for validity because we provided an initial function.
    stat_descriptors_[stat_id]->update_function(stats_[stat_id], user_value);
  }
}

void StatAggregator::Reset()
{
  // Ignore this call if recording
  if(state_ != STOPPED)
    return;
  
  hardcoded_stats_.Reset();
  
  // Reset non-hardcoded stats
  for(unsigned int s = 0; s < stat_descriptor_count_; s++) {
    // We will not check for validity because we provided an initial function.
    stat_descriptors_[s]->init_function(stats_[s]);
  }
}

void StatAggregator::StartRecording()
{
  // Ignore this call if already recording
  if(state_ != STOPPED)
    return;
  
  // Record start time
  start_time_ = std::chrono::system_clock::now();
  
  state_ = RECORDING;
}

void StatAggregator::StopRecording()
{
  // Ignore this call if not recording
  if(state_ != RECORDING)
    return;
  
  // Record stopping time
  auto stop_time = std::chrono::system_clock::now();
  
  // Update elapsed time
  std::chrono::duration<double> t_diff = stop_time - start_time_;
  hardcoded_stats_.seconds_elapsed += t_diff.count();
  
  state_ = STOPPED;
}

void StatAggregator::Snapshot() {
  // Ignore this call if not recording
  if (state_ != RECORDING)
    return;

  StopRecording();
  Generate();
  Reset();
  StartRecording();
}
  
void StatAggregator::SetCurrentExperiment(std::string current_experiment) {
  // Only change experiment name when not recording and already initialized
  if(state_!=STOPPED)
    return;
  
  // Call all StatSinks' SetCurrentExperiment method
  for(unsigned int s = 0; s < stat_sink_count_; s++) {
    stat_sinks_[s]->SetCurrentExperiment(current_experiment);
  }
}

void StatAggregator::SetCurrentTestingDataset(unsigned int current_testing_dataset) {
  hardcoded_stats_.current_testing_dataset = current_testing_dataset;
   // Only change dataset name when not recording and already initialized
  if(state_!=STOPPED)
    return;

  // Call all StatSinks' SetCurrentExperiment method
  for(unsigned int s = 0; s < stat_sink_count_; s++) {
    stat_sinks_[s]->SetCurrentTestingDataset(current_testing_dataset);
  }
}
  
}