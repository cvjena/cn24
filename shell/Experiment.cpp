/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ShellState.h"

namespace Conv {

CN24_SHELL_FUNC_IMPL(ExperimentBegin) {
  CN24_SHELL_FUNC_DESCRIPTION("Starts recording statistics to a specified file");

  char* file = nullptr;
  cargo_add_option(cargo, (cargo_option_flags_t)0, "file", "CSV file to record the statistics to",
    "s", &file);

  CN24_SHELL_PARSE_ARGS;

  // Check if shell state allows for model loading
  if(state_ != NET_AND_TRAINER_LOADED) {
    LOGERROR << "Cannot start experiment, no net is loaded or is loaded for prediction only.";
    return FAILURE;
  }

  if(stat_sinks_.size() > 1) {
    LOGERROR << "Cannot start experiment while experiment is still running!";
    return FAILURE;
  }

  std::string file_str = std::string(file);

  // Create CSVStatSink for experiment
  CSVStatSink* sink = new CSVStatSink(file_str);
  stat_sinks_.push_back(sink);
  System::stat_aggregator->RegisterSink(sink);

  // Set experiment name
  System::stat_aggregator->SetCurrentExperiment(file_str);

  return SUCCESS;
}


}
