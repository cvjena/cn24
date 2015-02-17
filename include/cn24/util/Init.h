/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file Init.h
 * @brief Provides initialization functions for several subsystems
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_INIT_H
#define CONV_INIT_H

namespace Conv {
class TensorViewer;
class System {
public:
  static void Init();
  static TensorViewer* viewer;
};
}

#endif
