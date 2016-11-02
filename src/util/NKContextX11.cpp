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
  screen_ = DefaultScreen(display_);
  window_ = XCreateSimpleWindow(display_, RootWindow(display_, screen_),
    0, 0, width_, height_, 1,
    BlackPixel(display_, screen_), WhitePixel(display_, screen_));
  XSelectInput(display_ , window_, ExposureMask | KeyPressMask);
  XMapWindow(display_, window_);
  
  XFont* font = nk_xfont_create(display_, "fixed");
  context_ = nk_xlib_init(font, display_, screen_, window_, width, height);
}

NKContext::~NKContext() {
  XCloseDisplay(display_);
}

void NKContext::ProcessEvents() {
  nk_input_begin(context_);
  XNextEvent(display_, &event_);
  if (XFilterEvent(&event_, window_)) {
    
  } else {
    nk_xlib_handle_event(display_, screen_, window_, &event_);
  }
  nk_input_end(context_);
}

void NKContext::Draw() {
  //XClearWindow(display_, window_);
  nk_xlib_render(window_, nk_rgb(30,30,30));
  XFlush(display_);
}
}

#endif

