/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CN24_NKCONTEXT_H
#define CN24_NKCONTEXT_H

#include "NKIncludes.h"
#include "../cn24/util/Tensor.h"

#ifdef BUILD_GUI_X11
extern "C" {
#include <X11/Xlib.h>
}

#include <chrono>
#endif

namespace Conv {

class NKContext;
class NKImage {
  friend class NKContext;
public:
  NKImage(NKContext& context, const Tensor& tensor, unsigned int sample);
  ~NKImage();
  void Update();
  void SetSample(unsigned int sample) {
    if(sample_ != sample) {
      sample_ = sample;
      Update();
    }
  }
  void* ptr();
  operator struct nk_image () const;
private:
  NKContext& context_;
  const Tensor& tensor_;
  unsigned int sample_;
#ifdef BUILD_GUI_X11
  XImage* image_;
  char* data_;
#endif
};

class NKContext {
  friend class NKImage;
public:
  NKContext(unsigned int width = 800, unsigned int height = 600);
  ~NKContext();
  
  void ProcessEvents();
  void Draw();
  nk_context* context_;
  operator nk_context* () const { return context_; }
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
