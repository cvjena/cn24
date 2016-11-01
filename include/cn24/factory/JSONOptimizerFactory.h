/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_JSONOPTIMIZERFACTORY_H
#define CONV_JSONOPTIMIZERFACTORY_H

#include "../util/JSONParsing.h"
#include "../math/Optimizer.h"

namespace Conv {
class JSONOptimizerFactory {
public:
  static Optimizer* ConstructOptimizer(JSON description);
};
}

#endif
