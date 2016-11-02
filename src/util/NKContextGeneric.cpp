/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef BUILD_GUI
#include "NKContext.h"
#include "Log.h"
namespace Conv {
NKContext::NKContext(unsigned int width, unsigned int height):
  width_(width), height_(height) {
  LOGERROR << "GUI not supported due to build options!";
}
}
#endif
