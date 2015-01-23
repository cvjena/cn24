/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file GradientTester.h
 * \brief Tests a net for gradient correctness
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_GRADIENTTESTER_H
#define CONV_GRADIENTTESTER_H

#include "Net.h"

namespace Conv {
class GradientTester {
public:
  static void TestGradient(Net& net);
};
  
  
}

#endif