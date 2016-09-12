/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_TEST_H
#define CONV_TEST_H

#include "Log.h"

namespace Conv {

template <typename T> void AssertEqual(T expected, T actual, std::string description) {
  if(expected != actual) {
    FATAL("Assertion failed: Expected " << description << " to be " << expected << ", actual value: " << actual);
  }
}

template <typename T> void AssertNotEqual(T expected, T actual, std::string description) {
  if(expected == actual) {
    FATAL("Assertion failed: Expected " << description << " to be " << expected << ", actual value: " << actual);
  }
}

template <typename T> void AssertLess(T expected, T actual, std::string description) {
  if(expected <= actual) {
    FATAL("Assertion failed: Expected " << description << " to be less than " << expected << ", actual value: " << actual);
  }
}


template <typename T> void AssertGreater(T expected, T actual, std::string description) {
  if(expected >= actual) {
    FATAL("Assertion failed: Expected " << description << " to be greater than " << expected << ", actual value: " << actual);
  }
}

template <typename T> void AssertGreaterEqual(T expected, T actual, std::string description) {
  if(expected > actual) {
    FATAL("Assertion failed: Expected " << description << " to be greater than " << expected << ", actual value: " << actual);
  }
}

template <typename T> void AssertLessEqual(T expected, T actual, std::string description) {
  if(expected < actual) {
    FATAL("Assertion failed: Expected " << description << " to be less than " << expected << ", actual value: " << actual);
  }
}

}

#endif
