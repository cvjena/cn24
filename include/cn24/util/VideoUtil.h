/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file VideoUtil.h
 * @class VideoUtil
 * @brief Loads Video files into a Tensor and writes them.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 *
 */

#ifndef CONV_VIDEOUTIL_H
#define CONV_VIDEOUTIL_H

#include <iostream>
#include <string>
#include <cstdio>

#include "Tensor.h"


namespace Conv {
struct VideoUtilPimpl;

class VideoUtil {
public:
  explicit VideoUtil(const std::string& filename);
  ~VideoUtil();

  unsigned int frames() const;
  unsigned int width() const;
  unsigned int height() const;

  bool ExtractFrame(unsigned int frame, Tensor& tensor, unsigned int sample);

  const std::string& last_error() const { return last_error_; }
private:
  VideoUtilPimpl* pimpl_;
  std::string last_error_ = "OK";
};


}
#endif
