/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file PNGLoader.h
 * \class PNGLoader
 * \brief Loads PNG files into a Tensor.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 *
 */

#ifndef CONV_JPGLOADER_H
#define CONV_JPGLOADER_H

#include <iostream>
#include <string>
#include <cstdio>

#ifdef BUILD_JPG
#include <jpeglib.h>
#endif

#include "Tensor.h"

namespace Conv {

class JPGLoader {
public:
  /**
   * \brief Loads a JPG file from an input stream into a Tensor.
   *
   * \param stream Input stream to read from
   * \param tensor Tensor to store the data in (will be resized, so
   *    you can use an empty Tensor)
   * \returns True on sucess, false otherwise
   */
  static bool LoadFromFile (const std::string& file, Tensor& tensor);
#ifdef BUILD_JPG
private:
  /** 
   * \brief Check if the provided stream contains a valid JPG file.
   *
   * \param stream Input stream to read from
   * \returns True if the file is a valid JPG file, false otherwise
   */
  static bool CheckSignature (std::istream& stream);
#endif
};


}
#endif
