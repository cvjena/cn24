/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file GradientTester.h
 * @brief Tests a net for gradient correctness
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_GRADIENTTESTER_H
#define CONV_GRADIENTTESTER_H

#include "../net/NetGraph.h"

namespace Conv {
class GradientTester {
public:
  /**
   * @brief Tests the gradients computed by the net numerically
   *
   * Only call this function on nets with a constant input, not a DatasetInputLayer!
   */
  static void TestGradient(NetGraph& net, unsigned int skip_weights = 0);
};
  
  
}

#endif
