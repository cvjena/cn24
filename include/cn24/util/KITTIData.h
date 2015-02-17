/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file KITTIData.h
 * @brief Helper class for fast KITTI data access
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_KITTIDATA_H
#define CONV_KITTIDATA_H

#include <string>
#include <iterator>

#include "Config.h"

namespace Conv {

// These are the 4 official KITTI benchmark categories
enum KITTICategory {
  KITTI_UM = 0,
  KITTI_UMM,
  KITTI_UU,
  KITTI_URBAN
};

class KITTIData {
public:
  KITTIData (std::string source);
  std::string getImage (KITTICategory category, int number, bool testing = false);
  std::string getRoadGroundtruth (KITTICategory category, int number);
  std::string getLaneGroundtruth (KITTICategory category, int number);
  std::string assembleFileName (KITTICategory category, int number, std::string infix = "");


  static datum LocalizedError(unsigned int x, unsigned int y, unsigned int w, unsigned int h);

private:
  std::string trainingImageFolder_;
  std::string trainingGroundtruthFolder_;
  std::string testingImageFolder_;
};

}


#endif

