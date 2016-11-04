/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifdef BUILD_GUI_X11
#include "Config.h"
#include "Tensor.h"
#include "NKIncludes.h"
#include "NKContext.h"

#include "Log.h"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <chrono>
#include <thread>

namespace Conv {
NKContext::NKContext(unsigned int width, unsigned int height):
  width_(width), height_(height) {
  LOGDEBUG << "Instance created.";

  // Open display
  display_ = XOpenDisplay(NULL);
  if(display_ == nullptr) {
    FATAL("Cannot open display!");
  }

  // Create window
  Window root = DefaultRootWindow(display_);
  screen_ = DefaultScreen(display_);
  visual_ = XDefaultVisual(display_, screen_);
  colormap_ = XCreateColormap(display_, root, visual_, AllocNone);

  attributes_.colormap = colormap_;
  attributes_.event_mask = ExposureMask | KeyPressMask | KeyReleaseMask |
                           ButtonPress | ButtonReleaseMask| ButtonMotionMask |
                           Button1MotionMask | Button3MotionMask | Button4MotionMask | Button5MotionMask|
                           PointerMotionMask | KeymapStateMask;

  window_ = XCreateWindow(display_, root, 0, 0, width_, height_, 0, XDefaultDepth(display_, screen_),
    InputOutput, visual_, CWEventMask | CWColormap, &attributes_);
  XStoreName(display_, window_, "CN24");
  XMapWindow(display_, window_);
  XWindowAttributes window_attributes;
  XGetWindowAttributes(display_, window_, &window_attributes);
  width_ = (unsigned int)window_attributes.width;
  height = (unsigned int)window_attributes.height;

  font_ = nk_xfont_create(display_, "fixed");
  context_ = nk_xlib_init(font_, display_, screen_, window_, width, height);

}

NKContext::~NKContext() {
  nk_xfont_del(display_, font_);
  nk_xlib_shutdown();
  XUnmapWindow(display_, window_);
  XFreeColormap(display_, colormap_);
  XDestroyWindow(display_, window_);
  XCloseDisplay(display_);
}

void NKContext::ProcessEvents() {
  nk_input_begin(context_);
  XEvent event;
  while(XCheckWindowEvent(display_, window_, attributes_.event_mask, &event)) {
    if(XFilterEvent(&event, window_))
      continue;
    nk_xlib_handle_event(display_, screen_, window_, &event);
  }

  nk_input_end(context_);
}

void NKContext::Draw() {
  auto current_time = std::chrono::system_clock::now();
  std::chrono::duration<double> t_diff = current_time - last_frame_;

  double frame_interval = 1.0/30.0;
  if(t_diff.count() < frame_interval) {
    std::this_thread::sleep_for(std::chrono::duration<double, std::milli>(frame_interval * 1000.0) - t_diff);
  }
  //XClearWindow(display_, window_);
  nk_xlib_render(window_, nk_rgb(30,30,30));
  XFlush(display_);
  last_frame_ = current_time;
}

NKImage::NKImage(NKContext &context, const Tensor &tensor, unsigned int sample) :
  context_(context), tensor_(tensor), sample_(sample) {
  data_ = new char[tensor_.width() * tensor_.height() * 4];
  image_ = XCreateImage(context_.display_, context_.visual_, 24, ZPixmap, 0, data_, tensor_.width(), tensor_.height(),
                        8, 0);
  for(unsigned int y = 0; y < tensor_.height(); y++) {
    for(unsigned int x = 0; x < tensor_.width(); x++) {
      const unsigned long R = UCHAR_FROM_DATUM(*tensor_.data_ptr(x, y, 0, sample_));
      const unsigned long G = UCHAR_FROM_DATUM(*tensor_.data_ptr(x, y, 1, sample_));
      const unsigned long B = UCHAR_FROM_DATUM(*tensor_.data_ptr(x, y, 2, sample_));
      unsigned long color = R << 16 | G << 8 | B;
      XPutPixel(image_, x, y, color);
    }
  }
}

NKImage::~NKImage() {
  XDestroyImage(image_);
  // delete[] data_; (X11 does this for us...)
}
void* NKImage::ptr() {
  return (void*)image_;
}
NKImage::operator struct nk_image() const {
  struct nk_image x = nk_image_ptr((void*)image_);
  x.h = tensor_.height();
  x.w = tensor_.width();
  return x;
}

void NKImage::Update() {
  for(unsigned int y = 0; y < tensor_.height(); y++) {
    for(unsigned int x = 0; x < tensor_.width(); x++) {
      const unsigned long R = UCHAR_FROM_DATUM(*tensor_.data_ptr(x, y, 0, sample_));
      const unsigned long G = UCHAR_FROM_DATUM(*tensor_.data_ptr(x, y, 1, sample_));
      const unsigned long B = UCHAR_FROM_DATUM(*tensor_.data_ptr(x, y, 2, sample_));
      unsigned long color = R << 16 | G << 8 | B;
      XPutPixel(image_, x, y, color);
    }
  }
}
}

#endif

