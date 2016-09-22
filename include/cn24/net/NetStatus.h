/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file NetStatus.h
 * @class NetStatus
 * @brief Informs layers and other clients about the network's training status
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_NETSTATUS_H
#define CONV_NETSTATUS_H

#include "../util/Init.h"
#include "../util/StatAggregator.h"

namespace Conv {

class NetStatus{
public:
	/**
   * @brief Returns true if the net is currently testing
   */
  inline bool IsTesting() const { return is_testing_; }

  /**
   * @brief Returs true if the net is currently gradient testing
   */
  inline bool IsGradientTesting() const { return is_gradient_testing_; }
  
  /**
   * @brief Sets this nets testing status
   * 
   * @param is_testing The new testing status
   */
  inline void SetIsTesting(bool is_testing) { 
    is_testing_ = is_testing;
    System::stat_aggregator->hardcoded_stats_.is_training = !is_testing;
  }

  /**
   * @brief Sets this nets gradient testing status
   *
   * @param is_gradient_testing The new gradient testing status
   */
  inline void SetIsGradientTesting(bool is_gradient_testing) {
    is_gradient_testing_ = is_gradient_testing;
  }
private:
	bool is_testing_ = false;
  bool is_gradient_testing_ = false;
};
}

#endif