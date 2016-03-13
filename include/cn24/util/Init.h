/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file Init.h
 * @brief Provides initialization functions for several subsystems and a
 *        singleton "System" class for global objects.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_INIT_H
#define CONV_INIT_H

#include <string>

namespace Conv {
class TensorViewer;
class StatAggregator;
class System {
public:
  static void Init(int requested_log_level = -1);
  static void Shutdown();
  static void GetExecutablePath(std::string& binary_path);
  static TensorViewer* viewer;
  static StatAggregator* stat_aggregator;
  static int log_level;
};
}

#endif
