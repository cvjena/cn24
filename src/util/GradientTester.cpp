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
  
void GradientTester::TestGradient ( NetGraph& graph ) {
  const double epsilon = 0.001;
  LOGDEBUG << "Testing gradient. FeedForward...";
	graph.FeedForward();
  LOGDEBUG << "Testing gradient. BackPropagate...";
  graph.BackPropagate();
  
	const datum initial_loss = graph.AggregateLoss();
  LOGDEBUG << "Initial loss: " << initial_loss;
  LOGDEBUG << "Using epsilon: " << epsilon;
  for(unsigned int l = 0; l < graph.GetNodes().size(); l++) {
		NetGraphNode* node = graph.GetNodes()[l];
		Layer* layer = node->layer;
    for(unsigned int p = 0; p < layer->parameters().size(); p++) {
			CombinedTensor* const param = layer->parameters()[p];
      LOGDEBUG << "Testing layer " << l << " (" << layer->GetLayerDescription() << "), parameter set " << p;
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
	
	param->data[e] = old_param + epsilon;
	graph.FeedForward();
	const double plus_loss = graph.AggregateLoss();
	
#ifdef BUILD_OPENCL
	param->data.MoveToCPU();
#endif
	param->data[e] = old_param - epsilon;
graph.FeedForward();
	const double minus_loss = graph.AggregateLoss();
	
	const double delta = param->delta[e];
	const double actual_delta = (plus_loss - minus_loss) / (2.0 * epsilon);
	
	const double ratio = actual_delta / delta;
	if(ratio > 1.02 || ratio < 0.98) {
	  if(ratio > 1.1 || ratio < 0.9) {
	    if(passed)
	      LOGWARN << "delta calculated: " << delta << ", actual: " << actual_delta << ", ratio: " << ratio;
	    passed = false;
	    std::cout << "!" << std::flush;
	  } else {
	  std::cout << "#" << std::flush;
	  }
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