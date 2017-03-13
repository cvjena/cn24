/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
/**
 * @file ShellState.h
 * @class ShellState
 * 
 * @brief ShellState encapsulates all NetGraph and Trainer ownership and
 * command processing for use in CN24-Shell
 */

#ifndef CONV_SHELLSTATE_H
#define CONV_SHELLSTATE_H

#include <cn24.h>

extern "C" {
  #include "../../../external/cargo/cargo.h"
}

#include <map>
#include <string>

namespace Conv {
  
class ShellState {
public:
  ShellState();
  
  /**
   * @brief CommandStatus encodes the return value of a command processing
   * operation
   */
  enum CommandStatus {
    SUCCESS = 0,
    FAILURE,
    WRONG_PARAMS,
    REQUEST_QUIT
  };
  
  /**
   * @brief Processes a single command line
   * @return CommandStatus that encodes success or failure
   */
  CommandStatus ProcessCommand(std::string);
  
  /*
   * The following macros are for the command table for cn24-shell
   */
#define CN24_SHELL_FUNC(func_name) CommandStatus func_name \
  (cargo_t cargo, int argc, char** argv, bool usage_only)
#define CN24_SHELL_FUNC_IMPL(func_name) ShellState::CommandStatus ShellState::func_name \
  (cargo_t cargo, int argc, char** argv, bool usage_only)
  
typedef CommandStatus (ShellState::*ShellFunction)(cargo_t cargo, int argc, char** argv, bool usage_only);
#define CN24_SHELL_CMD(cmd_name, func_name) {cmd_name, &ShellState::func_name}

#define CN24_SHELL_PARSE_ARGS {\
  if(usage_only) { std::cout << description << std::endl; return SUCCESS;} \
  if(cargo_parse(cargo, (cargo_flags_t)0, 1, argc, argv)) { \
    return WRONG_PARAMS; \
  }}

#define CN24_SHELL_FUNC_DESCRIPTION(desc) cargo_set_description(cargo, desc);\
  std::string description = desc;

  /*
   * Shell function definitions
   */
  CN24_SHELL_FUNC(Quit);
  CN24_SHELL_FUNC(CommandHelp);

  CN24_SHELL_FUNC(NetworkLoad);
  CN24_SHELL_FUNC(NetworkStatus);
  CN24_SHELL_FUNC(NetworkUnload);

  CN24_SHELL_FUNC(DataList);
  CN24_SHELL_FUNC(BundleLoad);
  CN24_SHELL_FUNC(BundleMove);
  CN24_SHELL_FUNC(SegmentMove);

  CN24_SHELL_FUNC(ModelLoad);
  CN24_SHELL_FUNC(ModelSave);

  CN24_SHELL_FUNC(ExperimentBegin);

  CN24_SHELL_FUNC(Train);
  CN24_SHELL_FUNC(Test);
  
  CN24_SHELL_FUNC(PredictImage);
private:
  Bundle* DataTakeBundle(const std::string& name);
  Bundle* DataFindBundle(const std::string& name);
  
  Segment* DataTakeSegment(const std::string& bundle_name,
    const std::string& segment_name);
  Segment* DataFindSegment(const std::string& bundle_name,
    const std::string& segment_name);

  void DataUpdated();
  /*
   * Shell command table
   */
  std::map<std::string, ShellFunction> cmd_name_func_map { 
    CN24_SHELL_CMD("quit", Quit),
    CN24_SHELL_CMD("help", CommandHelp),

    CN24_SHELL_CMD("net-load", NetworkLoad),
    CN24_SHELL_CMD("net-unload", NetworkUnload),
    CN24_SHELL_CMD("net-status", NetworkStatus),

    CN24_SHELL_CMD("data-list", DataList),
    CN24_SHELL_CMD("bundle-load", BundleLoad),
    CN24_SHELL_CMD("bundle-move", BundleMove),
    CN24_SHELL_CMD("segment-move", SegmentMove),

    CN24_SHELL_CMD("model-load", ModelLoad),
    CN24_SHELL_CMD("model-save", ModelSave),

    CN24_SHELL_CMD("experiment-begin", ExperimentBegin),

    CN24_SHELL_CMD("train", Train),
    CN24_SHELL_CMD("test", Test),

    CN24_SHELL_CMD("predict-image", PredictImage)
  };
  
private:
  enum State {
    NOTHING,
    NET_LOADED,
    NET_AND_TRAINER_LOADED
  };
  
  State state_ = NOTHING;
  
  ClassManager* class_manager_ = nullptr;
  NetGraph* graph_ = nullptr;
  Trainer* trainer_ = nullptr;
  BundleInputLayer* input_layer_ = nullptr;
  std::vector<StatSink*> stat_sinks_;
  
  std::vector<Bundle*>* training_bundles_ = new std::vector<Bundle*>();
  std::vector<datum>* training_weights_ = new std::vector<datum>();
  std::vector<Bundle*>* staging_bundles_ = new std::vector<Bundle*>();
  std::vector<Bundle*>* testing_bundles_ = new std::vector<Bundle*>();


  int global_random_seed = 19108128;
  
public:
  inline State state() { return state_; }
  inline NetGraph* graph() { return graph_; }
  inline BundleInputLayer* input_layer() { return input_layer_; }
  inline CombinedTensor* net_output() { return graph_->GetDefaultOutputNode()
    ->output_buffers[0].combined_tensor; }
};
}

#endif