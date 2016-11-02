/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifdef BUILD_GUI_X11
#include "NKIncludes.h"
#include "NKContext.h"

#include "Log.h"

#include <X11/Xlib.h>

namespace Conv {
NKContext::NKContext(unsigned int width, unsigned int height):
  width_(width), height_(height) {
  LOGDEBUG << "Instance created.";
  display_ = XOpenDisplay(NULL);
  if(display_ == nullptr) {
    FATAL("Cannot open display!");
  }
}
}

#endif

