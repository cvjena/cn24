/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file UIInfo.cpp
 * @brief Small test application for the UI library
 * 
 * @author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#include <cn24.h>
#include "private/NKContext.h"

int main() {
  Conv::System::Init(3);

  uint8_t* test_image = new uint8_t[400*400*3];
  for(int y = 0; y < 400; y++) {
    for (int x = 0; x < 400; x++) {
      test_image[3 * (y * 400 + x)] = x / 2;
      test_image[3 * (y * 400 + x) + 1] = x / 2;
      test_image[3 * (y * 400 + x) + 2] = x / 2;
    }
  }
  {
    Conv::NKContext context(1200, 800);
    bool running = true;
    while (running) {
      context.ProcessEvents();

      if(nk_begin(context, "UI Information", nk_rect(0,0,300,200),
      NK_WINDOW_BORDER | NK_WINDOW_MOVABLE | NK_WINDOW_SCALABLE | NK_WINDOW_CLOSABLE |
      NK_WINDOW_TITLE)) {
        nk_layout_row_dynamic(context, 30, 1);
        if(nk_button_label(context, "Exit")) {
          break;
        }

        nk_layout_row_dynamic(context, 400, 1);
        struct nk_image demo_image = nk_image_ptr(test_image);
        demo_image.w = 400;
        demo_image.h = 400;
        nk_image(context, demo_image);
      }
      nk_end(context);

      if(nk_window_is_closed(context, "UI Information"))
        break;
      context.Draw();
    }
  }
  delete[] test_image;
  LOGEND;
  
  return 0;
}
