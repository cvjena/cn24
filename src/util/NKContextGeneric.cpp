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
NKContext::~NKContext() {

}
void NKContext::ProcessEvents() {
  
}
void NKContext::Draw() {
  
}

NKImage::NKImage(NKContext &context, const Tensor &tensor, unsigned int sample) :
  context_(context), tensor_(tensor), sample_(sample) {

}
NKImage::~NKImage() {

}
void* NKImage::ptr() {
  return nullptr;
}
void NKImage::Update() {

}
NKImage::operator struct nk_image() const {
  struct nk_image t;
  return t;
}
}
#endif
