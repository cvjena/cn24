/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "StatAggregator.h"

#include <chrono>

namespace Conv {

void StatAggregator::Update(unsigned int stat_id)
{
  // Ignore this call if not recording
  if(!is_recording_)
    return;
  
  if(stat_id < stat_descriptor_count_) {
    if(stat_descriptors_[stat_id]->update_function) {
      stat_descriptors_[stat_id]->update_function(stats_[stat_id]);
    }
  }
}

void StatAggregator::Reset()
{
  // Ignore this call if recording
  if(is_recording_)
    return;
  
  hardcoded_stats_.Reset();
}

void StatAggregator::StartRecording()
{
  // Ignore this call if already recording
  if(is_recording_)
    return;
  
  // Record start time
  start_time_ = std::chrono::system_clock::now();
}

void StatAggregator::StopRecording()
{
  // Ignore this call if not recording
  if(!is_recording_)
    return;
  
  // Record stopping time
  auto stop_time = std::chrono::system_clock::now();
  
  // Update elapsed time
  std::chrono::duration<double> t_diff = stop_time - start_time_;
  hardcoded_stats_.seconds_elapsed += t_diff.count();
}

}