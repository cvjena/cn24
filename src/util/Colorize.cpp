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

#include <limits>

namespace Conv {

void Dataset::Colorize(Tensor& net_output_tensor, Tensor& target_tensor) {
#ifdef BUILD_OPENCL
  net_output_tensor.MoveToCPU();
  target_tensor.MoveToCPU();
#endif
  if(class_manager_->GetClassCount() == 1) {
    const unsigned int foreground_color = class_manager_->begin()->second.color;
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
    for(unsigned sample = 0; sample < net_output_tensor.samples(); sample++) {
       for(unsigned int y = 0; y < net_output_tensor.height(); y++) {
	for(unsigned int x = 0; x < net_output_tensor.width(); x++) {
	  unsigned int maxclass = 0;
	  datum maxvalue = std::numeric_limits<datum>::min();
		for(ClassManager::const_iterator it = class_manager_->begin(); it != class_manager_->end(); it++) {
      const unsigned int c = it->second.id;
	    const datum value = *net_output_tensor.data_ptr_const(x,y,c,sample);
	    if(value > maxvalue) {
	      maxvalue = value;
	      maxclass = c;
	    }
	  }
	  
	  const unsigned int foreground_color = class_manager_->GetClassInfoById(maxclass).second.color;
	  const datum r = DATUM_FROM_UCHAR((foreground_color >> 16) & 0xFF),
	  g = DATUM_FROM_UCHAR((foreground_color >> 8) & 0xFF),
	  b = DATUM_FROM_UCHAR(foreground_color & 0xFF);
	  
	  *target_tensor.data_ptr(x,y,0,sample) = r; // * maxvalue;
	  *target_tensor.data_ptr(x,y,1,sample) = g; // * maxvalue;
	  *target_tensor.data_ptr(x,y,2,sample) = b; // * maxvalue;
	}
      }     
    }
  }
}

}