/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CN24_NKCONTEXT_H
#define CN24_NKCONTEXT_H

#include "NKIncludes.h"

#ifdef BUILD_GUI_X11
extern "C" {
#include <X11/Xlib.h>
}
#endif

namespace Conv {
class NKContext {
public:
  NKContext(unsigned int width = 800, unsigned int height = 600);
private:
  unsigned int width_ = 0;
  unsigned int height_ = 0;
#ifdef BUILD_GUI_X11
  Display* display_;
  Window window_;
  XEvent event_;
#endif
};
}

#endif
