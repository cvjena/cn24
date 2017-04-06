/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

#include <iostream>
#include <string>
#include <cstdio>
#include <cstring>

#ifdef BUILD_OPENCV
#include <opencv2/opencv.hpp>

#endif

#include "Config.h"
#include "Log.h"
#include "Tensor.h"
#include "VideoUtil.h"

namespace Conv {

struct VideoUtilPimpl {
  unsigned int width, height, frames;
#ifdef BUILD_OPENCV
  cv::VideoCapture capture;
  VideoUtilPimpl(const std::string& filename) : capture(filename), width(0), height(0), frames(0) {}
#else
  VideoUtilPimpl(const std::string& filename) : width(0), height(0), frames(0) {}
#endif
};
VideoUtil::VideoUtil(const std::string& filename) : filename_(filename) {
  pimpl_ = new VideoUtilPimpl(filename);
#ifdef BUILD_OPENCV
  if (!pimpl_->capture.isOpened()) {
    last_error_ = "Could not open capture device.";
    return;
  }

  // Read capture metadata
  pimpl_->frames = (unsigned int) pimpl_->capture.get(CV_CAP_PROP_FRAME_COUNT);
  pimpl_->width = (unsigned int) pimpl_->capture.get(CV_CAP_PROP_FRAME_WIDTH);
  pimpl_->height = (unsigned int) pimpl_->capture.get(CV_CAP_PROP_FRAME_HEIGHT);

#else
  last_error_ = "Not compiled with opencv support!";
#endif
}

VideoUtil::~VideoUtil() {
  delete pimpl_;
}

bool VideoUtil::ExtractFrame(unsigned int frame, uint8_t* data) {
#ifdef BUILD_OPENCV
  if(frame >= pimpl_->frames) {
    last_error_ = "Requested frame out of bounds";
    return false;
  }

  // Seek to frame
  pimpl_->capture.set(CV_CAP_PROP_POS_FRAMES, (double)frame);

  // Check for null pointer
  if(data == nullptr) {
    last_error_ = "Null pointer for target data supplied!";
    return false;
  }

  // Get frame
  cv::Mat mat;
  pimpl_->capture >> mat;

  // Copy to image
  std::memcpy(data, mat.data, sizeof(uint8_t) * 3 * pimpl_->width * pimpl_->height);

  return true;
#else
  last_error_ = "OpenCV support was not compiled into CN24.";
  return false;
#endif
}

bool VideoUtil::ExtractFrame(unsigned int frame, Tensor &tensor, unsigned int sample) {
#ifdef BUILD_OPENCV
  if(frame >= pimpl_->frames) {
    last_error_ = "Requested frame out of bounds";
    return false;
  }

  // Seek to frame
  pimpl_->capture.set(CV_CAP_PROP_POS_FRAMES, (double)frame);

  // Check tensor dimensions
  if(tensor.width() != pimpl_->width || tensor.height() != pimpl_->height
    || tensor.maps() != 3) {
    if(tensor.elements() == 0) {
      // Resize!
      tensor.Resize(sample + 1, pimpl_->width, pimpl_->height, 3);
    } else {
      last_error_ = "Resizing would lead to data loss!";
      return false;
    }
  }

  // Check tensor sample count
  if(tensor.samples() <= sample) {
    tensor.Extend(sample + 1);
  }

  // Get frame
  cv::Mat mat;
  pimpl_->capture >> mat;

  // Copy to tensor
#ifdef BUILD_OPENCL
  tensor.MoveToCPU();
#endif

#pragma omp parallel for default(shared)
  for(unsigned int y = 0; y < pimpl_->height; y++) {
    const unsigned char* ptr = mat.ptr<unsigned char>(y);
    for(unsigned int x = 0; x < pimpl_->width; x++) {
      for(unsigned int c = 0; c < 3; c++) {
        unsigned int actual_channel = 2 - c;
        *(tensor.data_ptr(x, y, c)) = DATUM_FROM_UCHAR(ptr[x*3+actual_channel]);
      }
    }
  }

  return true;
#else
  last_error_ = "OpenCV support was not compiled into CN24.";
  return false;
#endif
}

unsigned int VideoUtil::frames() const { return pimpl_->frames; }
unsigned int VideoUtil::height() const { return pimpl_->height; }
unsigned int VideoUtil::width() const { return pimpl_->width; }

}
