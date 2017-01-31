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
#include "ClassManager.h"

namespace Conv {
typedef std::vector<BoundingBox> DetectionMetadata;
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
  datum score = 0;

  static bool CopyDetectionSample(
    JSON& sample,
    unsigned int target_index,
    Tensor* data,
    DetectionMetadataPointer metadata,
    ClassManager& class_manager,
    CopyMode copy_mode = NEVER_RESIZE);

  static bool CopyDetectionMetadata(
    JSON& sample,
    unsigned int image_width,
    unsigned int image_height,
    ClassManager& class_manager,
    DetectionMetadataPointer metadata
  );

  unsigned int GetSampleCount() const { return (unsigned int)samples_.size(); }
  JSON& GetSample(unsigned int index) {return samples_[index]; }

  JSON Serialize();
  bool Deserialize(
    JSON segment_descriptor,
    std::string folder_hint,
    int range_begin = 0,
    int range_end = -1);
  bool AddSample(
    JSON sample_descriptor,
    std::string folder_hint = {},
    bool use_rpath = false);

  bool RenameClass(const std::string& org_name, const std::string new_name);
private:
  std::vector<JSON> samples_;
  std::string last_folder_hint_ = {};
};

}
#endif
