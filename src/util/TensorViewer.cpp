/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#ifdef BUILD_GUI

#include <gtk/gtk.h>
#include "Config.h"
#include "Log.h"

#include "TensorViewer.h"

#include <thread>

namespace Conv {

TensorViewer::TensorViewer () {
  LOGDEBUG << "Instance created.";
}

void TensorViewer::show ( Tensor* tensor, const std::string& title, bool autoclose,unsigned int map, unsigned int sample ) {
  GtkWidget* window;
  window = gtk_window_new ( GTK_WINDOW_TOPLEVEL );
  gtk_window_set_title ( GTK_WINDOW ( window ), title.c_str() );
  g_signal_connect ( window, "destroy", G_CALLBACK ( gtk_main_quit ), NULL );

  GdkPixbuf* pixel_buffer = gdk_pixbuf_new ( GDK_COLORSPACE_RGB, gtk_false(), 8, tensor->width(), tensor->height() );
  copy ( tensor, pixel_buffer, map, sample, 1 );
  gtk_container_add ( GTK_CONTAINER ( window ), gtk_image_new_from_pixbuf ( pixel_buffer ) );

  gtk_widget_show_all ( window );

  auto closer = [&] () {
    std::this_thread::sleep_for ( std::chrono::milliseconds ( 300 ) );

    if ( autoclose ) {
      gtk_window_close(GTK_WINDOW(window));
      gtk_main_quit();
    }
  };
  std::thread t1 ( closer );

  gtk_main();

  t1.join();
}

void TensorViewer::copy ( Tensor* tensor, GdkPixbuf* targetb, unsigned int amap, unsigned int sample, datum factor ) {
  guchar* target = gdk_pixbuf_get_pixels ( targetb );
  unsigned int row_stride = gdk_pixbuf_get_rowstride ( targetb );

  if ( tensor->maps() == 3 ) {
    for ( unsigned int cmap = 0; cmap < 3; cmap++ ) {
      for ( unsigned int y = 0; y < tensor->height(); y++ ) {
        const Conv::datum* row = tensor->data_ptr_const ( 0, y, cmap, sample );
        guchar* target_row = &target[row_stride * y];

        for ( unsigned int x = 0; x < tensor->width(); x++ ) {
          target_row[ ( 3*x ) + cmap] = MCHAR_FROM_DATUM ( factor * row[x] );
        }
      }
    }
  } else {
    for ( unsigned int cmap = 0; cmap < 3; cmap++ ) {
      for ( unsigned int y = 0; y < tensor->height(); y++ ) {
        const Conv::datum* row = tensor->data_ptr_const ( 0, y, amap, sample );
        guchar* target_row = &target[row_stride * y];

        for ( unsigned int x = 0; x < tensor->width(); x++ ) {
          target_row[ ( 3*x ) + cmap] = MCHAR_FROM_DATUM ( factor * row[x] );
        }
      }
    }
  }
}


}

#else
namespace Conv {
  static int TensorViewerDummy = 0;
}
#endif
