/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_SHELLSTATE_H
#define CONV_SHELLSTATE_H

#include <cn24.h>

#include <string>

namespace Conv {
class ShellState {
public:
  void ProcessCommand(std::string);
};
}

#endif