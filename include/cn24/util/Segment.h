/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_SEGMENT_H
#define CONV_SEGMENT_H

#include <vector>
#include <string>

#include "BoundingBox.h"
#include "Tensor.h"
#include "JSONParsing.h"

namespace Conv {
typedef std::vector<BoundingBox>* DetectionMetadataPointer;

class Segment {
public:
  enum CopyMode {
    NEVER_RESIZE,
    CROP,
    SCALE
  };

  explicit Segment(std::string name) : name(name) {}
  std::string name;

  static bool CopyDetectionSample(
    JSON sample,
    unsigned int target_index,
    Tensor* data,
    DetectionMetadataPointer metadata,
    CopyMode copy_mode = NEVER_RESIZE);

  unsigned int GetSampleCount() { return (unsigned int)samples_.size(); }
  JSON GetSample( unsigned int index) {return samples_[index]; }

  JSON Serialize();
  bool Deserialize(
    JSON segment_descriptor,
    std::string folder_hint,
    int range_begin = 0,
    int range_end = -1);
  bool AddSample(
    JSON sample_descriptor,
    std::string folder_hint = {});
private:
  std::vector<JSON> samples_;
};

}
#endif
