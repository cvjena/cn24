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

#include <chrono>
#endif

namespace Conv {
class NKContext {
public:
  NKContext(unsigned int width = 800, unsigned int height = 600);
  ~NKContext();
  
  void ProcessEvents();
  void Draw();
#ifdef BUILD_GUI
  nk_context* context_;
  operator nk_context* () const { return context_; }
#endif
private:
  unsigned int width_ = 0;
  unsigned int height_ = 0;



#ifdef BUILD_GUI_X11
  Display* display_;
  Window window_;
  Colormap colormap_;
  XSetWindowAttributes attributes_;
  Visual* visual_;
  XFont* font_;
  int screen_;
  std::chrono::time_point<std::chrono::system_clock> last_frame_;
#endif
};
}

#endif
