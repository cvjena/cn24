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

namespace Conv {

class NetStatus{
public:
	/**
   * @brief Returns true if the net is currently testing
   */
  inline bool IsTesting() const { return is_testing_; } 
  
  /**
   * @brief Sets this net's testing status
   * 
   * @param is_testing The new testing status
   */
  inline void SetIsTesting(bool is_testing) { is_testing_ = is_testing; }
private:
	bool is_testing_ = false;
};
}

#endif