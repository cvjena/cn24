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
  Conv::NKContext context(1200, 800);
  
  LOGINFO << "Context at " << &context;
  
  while(1) {
    context.ProcessEvents();
    // ...
    context.Draw();
  }
  LOGEND;
  
  return 0;
}
