/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Segment.h"

namespace Conv {

bool Segment::CopyDetectionSample(JSON sample, unsigned int target_index, Tensor *data,
                                  DetectionMetadataPointer metadata, CopyMode copy_mode) {
  return false;
}

JSON Segment::Serialize() {
  return JSON::object();
}

bool Segment::Deserialize(JSON segment_descriptor, std::string folder_hint, int range_begin, int range_end) {
  return false;
}

bool Segment::AddSample(JSON sample_descriptor, std::string folder_hint) {
  return false;
}
}
