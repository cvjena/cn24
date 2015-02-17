/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file JPGUtil.h
 * @class JPGUtil
 * @brief Loads JPG files into a Tensor and writes them.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CONV_JPGLOADER_H
#define CONV_JPGLOADER_H

#include <iostream>
#include <string>
#include <cstdio>

#include "Tensor.h"

namespace Conv {

class JPGUtil {
public:
  /**
   * @brief Loads a JPG file from an input stream into a Tensor.
   *
   * @param file Input file to read from
   * @param tensor Tensor to store the data in (will be resized, so
   *    you can use an empty Tensor)
   * @returns True on sucess, false otherwise
   */
  static bool LoadFromFile (const std::string& file, Tensor& tensor);
  
  /**
   * @brief Writes a Tensor to an output stream in PNG format.
   * 
   * @param file Output file to write to
   * @param tensor Tensor to read the data from (must be 1 sample,
   * 	3 maps)
   * @returns True on success, false otherwise
   */
  static bool WriteToFile (const std::string& file, Tensor& tensor);
private:
  /** 
   * @brief Check if the provided stream contains a valid JPG file.
   *
   * @param stream Input stream to read from
   * @returns True if the file is a valid JPG file, false otherwise
   */
  static bool CheckSignature (std::istream& stream);
};


}
#endif
