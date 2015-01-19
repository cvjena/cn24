/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testTensor.cpp
 * \brief Test application for Tensor.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <cn24.h>

int main() {

  Conv::Tensor tensor (4, 10, 20, 3);
  Conv::Tensor second_tensor;

  LOGINFO << "Offset of (0,0,0,0): " << tensor.Offset (0, 0, 0, 0);
  LOGINFO << "Offset of (1,0,0,0): " << tensor.Offset (1, 0, 0, 0);
  LOGINFO << "Offset of (0,1,0,0): " << tensor.Offset (0, 1, 0, 0);
  LOGINFO << "Offset of (0,0,1,0): " << tensor.Offset (0, 0, 1, 0);
  LOGINFO << "Offset of (0,0,0,1): " << tensor.Offset (0, 0, 0, 1);

  LOGINFO << "Result of reshaping: " << tensor.Reshape (12, 10, 20, 1);
  LOGINFO << "Result of reshaping: " << second_tensor.Reshape (12, 10, 20, 1);

  LOGINFO << "Matching size...";

  second_tensor.Resize (tensor);

  LOGINFO << "Result of reshaping: " << second_tensor.Reshape (12, 10, 20, 1);
  LOGINFO << "Elements before moving: " << tensor.elements() <<
          ", 0";
  Conv::Tensor third_tensor (std::move (tensor));
  LOGINFO << "Elements after moving: " << tensor.elements() <<
          ", " << third_tensor.elements();

  LOGINFO << "Test: " << third_tensor << ", old: " << tensor;
  
  second_tensor.Resize(800, 3, 2, 1);
  std::ofstream file("tmp.Tensor", std::ios::out | std::ios::binary);
  
  // More than one Tensor per file
  tensor.Serialize(file);
  second_tensor.Serialize(file);
  third_tensor.Serialize(file);
  
  file.close();
  
  Conv::Tensor file_tensor;
  std::ifstream file_input("tmp.Tensor", std::ios::in | std::ios::out);
  
  tensor.Deserialize(file_input);
  second_tensor.Deserialize(file_input);
  file_tensor.Deserialize(file_input);
  
  LOGINFO << "Deserialized: " << file_tensor;
  
  for(std::size_t e = 0; e < file_tensor.elements(); e++) {
    if(third_tensor(e) != file_tensor(e)) {
      LOGERROR << "Tensor has changed!";
      LOGEND;
      return -1;
    }
  }
  
  file_input.close();
  
  LOGEND;

}
