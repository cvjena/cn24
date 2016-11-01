/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */


#include "Optimizer.h"
#include "SGDOptimizer.h"
#include "AdamOptimizer.h"

#include "JSONOptimizerFactory.h"

namespace Conv {
Optimizer* JSONOptimizerFactory::ConstructOptimizer(JSON descriptor) {
  if(descriptor.count("optimization_method") == 1 && descriptor["optimization_method"].is_string()) {
    std::string method = descriptor["optimization_method"];
    if(method.compare("gd") == 0) {
      return new SGDOptimizer(descriptor);
    } else if(method.compare("adam") == 0) {
      return new AdamOptimizer(descriptor);
    }
    return nullptr;
  } else
    return nullptr;
}

}
