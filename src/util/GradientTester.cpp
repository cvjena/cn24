/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include "Log.h"
#include "Net.h"
#include "GradientTester.h"

namespace Conv {
  
void GradientTester::TestGradient ( Net& net ) {
  const datum epsilon = 0.001;
  LOGDEBUG << "Testing gradient. FeedForward...";
  net.FeedForward();
  LOGDEBUG << "Testing gradient. BackPropagate...";
  net.BackPropagate();
  
  const datum initial_loss = net.lossfunction_layer()->CalculateLossFunction();
  LOGDEBUG << "Initial loss: " << initial_loss;
  LOGDEBUG << "Using epsilon: " << epsilon;
  for(unsigned int l = 0; l < net.layers_.size(); l++) {
    for(unsigned int p = 0; p < net.layers_[l]->parameters().size(); p++) {
      CombinedTensor* const param = net.layers_[l]->parameters_[p];
      LOGDEBUG << "Testing layer " << l << ", parameter set " << p;
      LOGDEBUG << param->data;
      bool passed = true;
      unsigned int okay = 0;
      for(unsigned int e = 0; e < param->data.elements(); e++)
      {
#ifdef BUILD_OPENCL
	param->data.MoveToCPU();
	param->delta.MoveToCPU();
#endif
	const datum old_param = param->data(e);
	param->data[e] += epsilon;
	const datum delta = epsilon * param->delta[e];
	net.FeedForward();
	const datum new_loss = net.lossfunction_layer()->CalculateLossFunction();
	const datum actual_delta = new_loss - initial_loss;
	
	const datum ratio = actual_delta / delta;
	if(ratio > 1.01 || ratio < 0.99) {
	  if(passed && (ratio > 1.1 || ratio < 0.9))
	    LOGWARN << "delta expected: " << delta << ",actual: " << actual_delta << ",ratio: " << ratio;
	  passed = false;
	  std::cout << "#" << std::flush;
	}
	else {
	  std::cout << "." << std::flush;
	  okay++;
	}
#ifdef BUILD_OPENCL
	param->data.MoveToCPU();
#endif
	param->data[e] = old_param;
      }
      std::cout << "\n";
      if(passed) {
	LOGINFO << "Okay!";
      } else {
	LOGERROR << "Failed!";
      }
      LOGINFO << okay << " of " << param->data.elements() << " gradients okay";
    }
  }
}

  
}