/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Segment.h"

#include "PathFinder.h"
#include "Log.h"

namespace Conv {

bool Segment::CopyDetectionSample(JSON sample, unsigned int target_index, Tensor *data,
                                  DetectionMetadataPointer metadata, ClassManager& class_manager,
                                  CopyMode copy_mode) {
  return false;
}

JSON Segment::Serialize() {
  return JSON::object();
}

bool Segment::Deserialize(JSON segment_descriptor, std::string folder_hint, int range_begin, int range_end) {
  bool success = true;
  if(segment_descriptor.count("samples") == 1 && segment_descriptor["samples"].is_array()) {
    if(range_end < 0)
      range_end = segment_descriptor["samples"].size() - 1;
    else if(range_end >= segment_descriptor["samples"].size()) {
      LOGWARN << "Segment \"" << name << "\": Descriptor only has " << segment_descriptor["samples"].size() << " samples, cannot set end of range to index " << range_end << "!";
      range_end = segment_descriptor["samples"].size() - 1;
    }
    if(range_begin < 0) {
      range_begin = 0;
    } else if(range_begin >= segment_descriptor["samples"].size()) {
      LOGWARN << "Segment \"" << name << "\": Descriptor only has " << segment_descriptor["samples"].size() << " samples, cannot set beginning of range to index " << range_begin << "!";
      range_begin = segment_descriptor["samples"].size() - 1;
    }
    for(unsigned int s = (unsigned int)range_begin; s <= (unsigned int)range_end; s++) {
      if(segment_descriptor["samples"][s].is_object()) {
        JSON sample_descriptor = segment_descriptor["samples"][s];
        success &= AddSample(sample_descriptor, folder_hint);
      } else {
        LOGWARN << "Segment \"" << name << "\": Not an object: " << segment_descriptor["samples"][s].dump() << ", skipping";
      }
    }
  }
  return success;
}

bool Segment::AddSample(JSON sample_descriptor, std::string folder_hint) {
  if(sample_descriptor.count("image_filename") == 1 && sample_descriptor["image_filename"].is_string()) {
    std::string image_filename = sample_descriptor["image_filename"];
    std::string resolved_path = PathFinder::FindPath(image_filename, folder_hint);
    if(resolved_path.length() > 0) {
      sample_descriptor["image_rpath"] = resolved_path;
      samples_.push_back(sample_descriptor);
      return  true;
    } else {
      LOGERROR << "Could not find sample \"" << image_filename << "\", skipping!";
      return false;
    }
  } else {
    LOGERROR << "Sample is missing image file name: " << sample_descriptor.dump();
    return false;
  }
}
}
