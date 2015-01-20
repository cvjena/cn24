/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file PNGUtil.h
 * \class PNGUtil
 * \brief Loads PNG files into a Tensor.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 *
 */

#ifndef CONV_PNGUTIL_H
#define CONV_PNGUTIL_H

#include <iostream>

#ifdef BUILD_PNG
#include <png.h>
#endif

#include "Tensor.h"

namespace Conv {

class PNGUtil {
public:
  /**
   * \brief Loads a PNG file from an input stream into a Tensor.
   *
   * \param stream Input stream to read from
   * \param tensor Tensor to store the data in (will be resized, so
   *    you can use an empty Tensor)
   * \returns True on sucess, false otherwise
   */
  static bool LoadFromStream (std::istream& stream, Tensor& tensor); 
  
  /**
   * \brief Writes a Tensor to an output stream in PNG format.
   * 
   * \param stream Output stream to write to
   * \param tensor Tensor to read the data from (must be 1 sample,
   * 	3 maps)
   * \returns True on success, false otherwise
   */
  static bool WriteToStream (std::ostream& stream, Tensor& tensor);
#ifdef BUILD_PNG
private:
  /** 
   * \brief Check if the provided stream contains a valid PNG file.
   *
   * \param stream Input stream to read from
   * \returns True if the file is a valid PNG file, false otherwise
   */
  static bool CheckSignature (std::istream& stream);

  /**
   * \brief This function is needed because libPNG doesn't support streams.
   */
  static void ReadFromStream (png_structp png_handle, png_bytep data,
                              png_size_t length);

#endif
};


}
#endif
