/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifdef BUILD_GUI_GTK
#include <gtk/gtk.h>
#include <string>
#include <thread>
#include <algorithm>
#include <sstream>
#include "Config.h"
#include "Tensor.h"
#endif

#include "Log.h"
#include "TensorViewer.h"
#include "NKContext.h"

namespace Conv {

#ifdef BUILD_GUI_GTK
void copyTV ( Tensor* tensor, GdkPixbuf* targetb, unsigned int amap, unsigned int sample, datum factor, unsigned int scale ) {
  guchar* target = gdk_pixbuf_get_pixels ( targetb );
  unsigned int row_stride = gdk_pixbuf_get_rowstride ( targetb );

  if ( tensor->maps() == 3 ) {
    for ( unsigned int cmap = 0; cmap < 3; cmap++ ) {
      for ( unsigned int y = 0; y < scale * tensor->height(); y++ ) {
        const Conv::datum* row = tensor->data_ptr_const ( 0, y/scale, cmap, sample );
        guchar* target_row = &target[row_stride * y];

        for ( unsigned int x = 0; x < scale * tensor->width(); x++ ) {
          target_row[ ( 3*x ) + cmap] = UCHAR_FROM_DATUM ( factor * row[x/scale] );
        }
      }
    }
  } else {
    for ( unsigned int cmap = 0; cmap < 3; cmap++ ) {
      for ( unsigned int y = 0; y < scale * tensor->height(); y++ ) {
        const Conv::datum* row = tensor->data_ptr_const ( 0, y/scale, amap, sample );
        guchar* target_row = &target[row_stride * y];

        for ( unsigned int x = 0; x < scale * tensor->width(); x++ ) {
          const datum value = std::max ( std::min ( factor * row[x/scale],1.0f ),-1.0f );
          target_row[ ( 3*x ) + cmap] =
            UCHAR_FROM_DATUM ( ( ( value < 0 && cmap == 0 ) || ( value >= 0 && cmap == 1 ) ) ? value : 0 );
        }
      }
    }
  }
}
#endif

TensorViewer::TensorViewer () {
  LOGDEBUG << "Instance created.";
}

void TensorViewer::show ( Tensor* tensor, const std::string& title, bool autoclose, unsigned int map, unsigned int sample ) {
  NKContext ctx{1280,800};
  NKImage tensor_image{ctx, *tensor, sample};
  bool running = true;
  while(running) {
    ctx.ProcessEvents();
    if (nk_begin(ctx, title.c_str(), nk_rect(0, 0, 800, 600), NK_WINDOW_TITLE | NK_WINDOW_CLOSABLE)) {
      nk_layout_row_dynamic(ctx, 30, 2);
      nk_value_uint(ctx, "Sample", sample);
      nk_value_uint(ctx, "Map", map);
      nk_layout_row_static(ctx, tensor->height(), tensor->width(), 1);
      nk_image(ctx, tensor_image);
    }
    nk_end(ctx);
    if(nk_window_is_closed(ctx, title.c_str()))
      running = false;
    ctx.Draw();
  }
#ifdef BUILD_GUI_GTK
  if(tensor->elements() == 0)
    return;

  unsigned int factor = 1;
  if(tensor->width() < 256 && tensor->height() < 256) {
    factor = 256/tensor->width();
  }
  GtkWidget* window;
  window = gtk_window_new ( GTK_WINDOW_TOPLEVEL );
  std::stringstream ss;
  ss << title << ": map " << map << "/" << tensor->maps() <<  " ,sample " << sample << "/" << tensor->samples();
  gtk_window_set_title ( GTK_WINDOW ( window ), ss.str().c_str() );
  g_signal_connect ( window, "destroy", G_CALLBACK ( gtk_main_quit ), NULL );

  GdkPixbuf* pixel_buffer = gdk_pixbuf_new ( GDK_COLORSPACE_RGB, gtk_false(), 8, factor * tensor->width(), factor * tensor->height() );
  copyTV ( tensor, pixel_buffer, map, sample, 1, factor );
  gtk_container_add ( GTK_CONTAINER ( window ), gtk_image_new_from_pixbuf ( pixel_buffer ) );

  gtk_widget_show_all ( window );

  auto closer = [&] () {
    std::this_thread::sleep_for ( std::chrono::milliseconds ( 300 ) );

    if ( autoclose ) {
      gtk_window_close ( GTK_WINDOW ( window ) );
      gtk_main_quit();
    }
  };
  std::thread t1 ( closer );

  gtk_main();

  t1.join();
#else
  UNREFERENCED_PARAMETER(autoclose);
  UNREFERENCED_PARAMETER(map);
  UNREFERENCED_PARAMETER(sample);
  LOGWARN << "Cannot show Tensor: " << tensor << ", " << title;
#endif
}

}
