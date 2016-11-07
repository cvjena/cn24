/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <fstream>
#include <sstream>

#include <iomanip>
#include <limits>

#include <cctype>
#include <string>
#include <functional>
#include <algorithm>

#include "CSVStatSink.h"

namespace Conv {
  
void CSVStatSink::Initialize(std::vector<StatDescriptor*>& stat_descriptors) {
  stat_descriptors_ = stat_descriptors;
}
  
void CSVStatSink::Process(Conv::HardcodedStats &hardcoded_stats, std::vector<Stat *> &stats) {
  if(csv_stream_ == nullptr)
    return;
  
  // Write hardcoded stats
  (*csv_stream_) << (hardcoded_stats.is_training ? "1" : "0") << ",";
  (*csv_stream_) << hardcoded_stats.epoch << ",";
  (*csv_stream_) << hardcoded_stats.iterations << ",";
  (*csv_stream_) << std::setprecision(std::numeric_limits<double>::digits10 + 1) << hardcoded_stats.seconds_elapsed << ",";
  (*csv_stream_) << current_dataset_ << ",";
  
  // Write values...
  for (unsigned int s = 0; s < stat_descriptors_.size(); s++) {
    // ...but only if not NULL
    if(!stats[s]->is_null)
      (*csv_stream_) << std::setprecision(std::numeric_limits<double>::digits10 + 1) << stats[s]->value;
    
    // Add comma except for last line
    if(s < (stat_descriptors_.size() - 1))
      (*csv_stream_) << ",";
  }
  (*csv_stream_) << "\n";
  (*csv_stream_) << std::flush;
  
}
  
bool isnalnum(char c) {
  return !std::isalnum((int)c);
}
  
void CSVStatSink::SetCurrentExperiment(std::string current_experiment) {
  // Close stream if already open
  if(csv_stream_ != nullptr) {
    csv_stream_->close();
    delete csv_stream_;
  }
  
  // Generate filename
  std::stringstream csv_filename_ss;
  csv_filename_ss << "csv/" << current_experiment << ".csv";
  std::string csv_filename=csv_filename_ss.str();
  
  // Open new stream
  csv_stream_ = new std::ofstream(csv_filename, std::ios::out);
  
  // Test if stream works
  if(!csv_stream_ ->good()) {
    LOGERROR << "Cannot open " << csv_filename << " for writing!";
    delete csv_stream_;
    csv_stream_ = nullptr;
  }
  // Write header for hardcoded stats
  (*csv_stream_) << "IsTraining,Epoch,Iterations,SecondsElapsed,TestingDataset,";
  
  // Write header for non-hardcoded stats
  for (unsigned int s = 0; s < stat_descriptors_.size(); s++) {
    std::string description = stat_descriptors_[s]->description;
    
    // Strip non-alphanumeric characters
    description.erase(std::remove_if(description.begin(), description.end(), isnalnum), description.end());
    (*csv_stream_) << description;
    if(s < (stat_descriptors_.size() - 1))
      (*csv_stream_) << ",";
  }
  (*csv_stream_) << "\n";
  (*csv_stream_) << std::flush;
}

void CSVStatSink::SetCurrentTestingDataset(unsigned int current_dataset) {
  current_dataset_ = current_dataset;
}
  
}