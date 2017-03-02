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
    REQUEST_QUIT
  };
  
  /**
   * @brief Processes a single command line
   * @return CommandStatus that encodes success or failure
   */
  CommandStatus ProcessCommand(std::string);
  
private:
  enum State {
    NOTHING,
    NET_LOADED,
    NET_AND_TRAINER_LOADED
  };
  
  NetGraph* graph_ = nullptr;
  Trainer* trainer_ = nullptr;
  SegmentSetInputLayer* input_layer_ = nullptr;
  
  
};
}

#endif