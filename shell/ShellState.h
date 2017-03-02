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
  #include "cargo.h"
}

#include <map>
#include <string>

namespace Conv {
  
class ShellState {
public:
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
  CN24_SHELL_FUNC(NetworkLoad);
  CN24_SHELL_FUNC(CommandHelp);
private:
  /*
   * Shell command table
   */
  std::map<std::string, ShellFunction> cmd_name_func_map { 
    CN24_SHELL_CMD("quit", Quit),
    CN24_SHELL_CMD("help", CommandHelp),
    CN24_SHELL_CMD("net-load", NetworkLoad)
  };
  
public:
  
  
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
  SegmentSetInputLayer* input_layer_ = nullptr;
};
}

#endif