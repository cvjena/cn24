/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Dataset.h"

#include "Log.h"
#include "Config.h"
#include "Tensor.h"

#ifdef BUILD_OPENMP
#include <omp.h>
#endif

namespace Conv {

void Dataset::Colorize(Tensor& net_output_tensor, Tensor& target_tensor) {
#ifdef BUILD_OPENCL
  net_output_tensor.MoveToCPU();
  target_tensor.MoveToCPU();
#endif
  if(GetClasses() == 1) {
    const unsigned int foreground_color = GetClassColors()[0];
    const datum r = DATUM_FROM_UCHAR((foreground_color >> 16) & 0xFF),
    g = DATUM_FROM_UCHAR((foreground_color >> 8) & 0xFF),
    b = DATUM_FROM_UCHAR(foreground_color & 0xFF);
    
    for(unsigned int sample = 0; sample < net_output_tensor.samples(); sample++) {
      for(unsigned int y = 0; y < net_output_tensor.height(); y++) {
	for(unsigned int x = 0; x < net_output_tensor.width(); x++) {
	  datum value = *net_output_tensor.data_ptr_const(x,y,0,sample);
	  value += 1.0;
	  value /= 2.0;
	  
	  *target_tensor.data_ptr(x,y,0,sample) = r * value;
	  *target_tensor.data_ptr(x,y,1,sample) = g * value;
	  *target_tensor.data_ptr(x,y,2,sample) = b * value;
	}
      }
    }
    
  } else {
    LOGWARN << "This code path is not yet implemented!";
  }
}

}