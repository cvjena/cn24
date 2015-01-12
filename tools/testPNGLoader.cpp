/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testPNGLoader.cpp
 * \brief Test application for PNGLoader.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#include <iostream>
#include <string>
#include <fstream>

#include <string>
#include <cn24.h>

#ifdef BUILD_GUI
#include <gtk/gtk.h>
#include <gdk-pixbuf/gdk-pixbuf.h>

void OnDestroy(GtkWidget* widget, gpointer user_data) {
  gtk_main_quit();
}
#endif

int main (int argc, char* argv[]) {
  // Check argument count
  if (argc != 2) {
    LOGERROR << "USAGE: " << std::string (argv[0]) << " <PNG file>";
    return -1;
  }

  std::string file_name (argv[1]);

  std::ifstream file_stream (file_name);

  if (!file_stream.good()) {
    LOGERROR << "Cannot open " << file_name << "!";
    return -1;
  }
  
  Conv::Tensor tensor;
  bool result = Conv::PNGLoader::LoadFromStream(file_stream, tensor);
  
  if (!result) {
    LOGERROR << file_name << " is not a PNG file!";
    return -1;
  }
  
  LOGINFO << "Width: " << tensor.width() << ", height: " << tensor.height();
  LOGINFO << "Channels: " << tensor.maps() << ", samples: " << tensor.samples();
  
#ifndef BUILD_GUI
  LOGINFO << "Sorry, no GUI support built in.";
#else
  gtk_init(&argc, &argv);
  GtkWidget* window = gtk_window_new(GTK_WINDOW_TOPLEVEL);
  g_signal_connect(window, "destroy", G_CALLBACK(OnDestroy), NULL);
  
  GdkPixbuf* pixel_buffer = gdk_pixbuf_new(GDK_COLORSPACE_RGB, gtk_false(), 8, tensor.width(), tensor.height());
  guchar* target = gdk_pixbuf_get_pixels(pixel_buffer);
  Conv::datum* tensor_data = tensor.data_ptr();
  for(std::size_t channel = 0; channel < tensor.maps() && channel < 3;
      channel++) {
    for(std::size_t y = 0; y < tensor.height(); y++) {
      Conv::datum* row = &tensor_data[tensor.Offset(0, y, channel,0)];
      guchar* target_row = &target[tensor.width() * 3 * y];
      for(std::size_t x = 0; x < tensor.width(); x++) {
        target_row[x * 3 + channel] = UCHAR_FROM_DATUM(row[x]);
      }
    }
  }
  
  gtk_container_add(GTK_CONTAINER(window), gtk_image_new_from_pixbuf(pixel_buffer));
  
  gtk_widget_show_all(window);
  gtk_main();
#endif
  
  LOGEND;
  return 0;

}
