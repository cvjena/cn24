/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testLocalizedError.cpp
 * \brief Runs a localized error function and outputs the results.
 * 
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <cstring>

#include <cn24.h>

int main (int argc, char* argv[]) {
  Conv::Tensor result(1,1226,370);
  Conv::datum sum = 0.0;
  for (unsigned int x = 0; x < 1225; x++) {
    for (unsigned int y = 0; y < 369; y++) {
      Conv::datum* ptr = result.data_ptr(x,y);
      const Conv::datum error = Conv::KITTIData::LocalizedError(x,y,1226,370);
      *ptr = 0.06 * error;
      sum += error;
    }
  }

  std::ofstream ofile ("locerror.png.data", std::ios::binary | std::ios::out);
  result.Serialize(ofile, true);
  
  LOGINFO << "Sum: " << sum << ", part: " << (1226.0*370.0)/sum;
  LOGEND;
  return 0;
}