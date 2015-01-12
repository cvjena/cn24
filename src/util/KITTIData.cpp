/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <string>
#include <iostream>
#include <sstream>
#include <iomanip>

#include <cmath>

#include "KITTIData.h"

namespace Conv {

KITTIData::KITTIData (std::string source) {
  // Look for subfolders
  trainingImageFolder_ = source + "training/image_2/";
  trainingGroundtruthFolder_ = source + "training/gt_image_2/";
  testingImageFolder_ = source + "testing/image_2/";
}

std::string KITTIData::getImage (KITTICategory category, int number,
                                    bool testing) {
  std::string imageFolder = trainingImageFolder_;
  if (testing)
    imageFolder = testingImageFolder_;

  std::string imageFileName = imageFolder + assembleFileName (category, number);

  return imageFileName;
  
}

std::string KITTIData::getRoadGroundtruth (KITTICategory category, int number) {
  std::string imageFolder = trainingGroundtruthFolder_;

  std::string imageFileName = imageFolder + assembleFileName (category, number, "road_");

  return imageFileName;
}

std::string KITTIData::getLaneGroundtruth (KITTICategory category, int number) {
  std::string imageFolder = trainingGroundtruthFolder_;

  std::string imageFileName = imageFolder + assembleFileName (category, number, "lane_");

  return imageFileName;
}

std::string KITTIData::assembleFileName (KITTICategory category, int number,
    std::string infix) {
  // Assemble file name
  std::string fileNamePrefix;

  switch (category) {
    case KITTI_UM:
      fileNamePrefix = "um_";
      break;
    case KITTI_UMM:
      fileNamePrefix = "umm_";
      break;
    case KITTI_UU:
      fileNamePrefix = "uu_";
      break;
    case KITTI_URBAN:
      fileNamePrefix = "urban_";
      break;
  }

  std::ostringstream fileNumber;
  fileNumber << std::setfill ('0') << std::right << std::setw (6) << number;

  std::string fileName = fileNamePrefix + infix + fileNumber.str() + ".png";

  return fileName;
}

datum KITTIData::LocalizedError (unsigned int x, unsigned int y) {
  const int distance = std::abs(613-(int)x);
  const datum xy_allowed = (distance >  (3 * ((int)y - 140))) ? 0.0 : 1.0;
  const datum y_factor = (y > 190) ? 1.0 : 0.0;
  
  const datum shape_weight = (xy_allowed * y_factor);
  
  const datum quotient = (y<=170) ? 7.0 :
    fmax(1.0,fmin(7.0, 200.0/((datum)y - 170.0)));
    
  const datum error = 0.25 + ((pow(quotient,2.0)-0.25) * shape_weight);
  
  return 0.51 * error;
}


}
