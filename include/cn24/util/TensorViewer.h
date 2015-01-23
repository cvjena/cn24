/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

/**
 * \file TensorViewer.h
 * \class TensorViewer
 * \brief A GTK+ interface to view Tensors.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CN24_TENSORVIEWER_H
#define CN24_TENSORVIEWER_H
#ifdef BUILD_GUI

#include "Tensor.h"
#include <string>
#include <gtk/gtk.h>

namespace Conv {

class TensorViewer {
public:
  TensorViewer();
  void show(Tensor* tensor, const std::string& title = "Tensor Viewer",bool autoclose = false, unsigned int map = 0, unsigned int sample = 0);
private:
  static void copy(Tensor* tensor, GdkPixbuf* targetb, unsigned int map, unsigned int sample, datum factor);
};


// We show this if GTK+ is not enabled in the build system
#else
#include "Tensor.h"
namespace Conv {
class TensorViewer {
public:
  TensorViewer() { LOGDEBUG << "Not showing TensorViewer"; };
  void show(Tensor* tensor, const std::string& title = "Tensor Viewer",bool autoclose = false, unsigned int map = 0, unsigned int sample = 0) {};
};

#endif
}
#endif