/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "SegmentSet.h"

#include "Segment.h"

namespace Conv {

bool SegmentSet::CopyDetectionSample(unsigned int source_index, unsigned int target_index, Tensor *data,
                                     DetectionMetadataPointer metadata, ClassManager &class_manager,
                                     Segment::CopyMode copy_mode) {
  JSON sample = GetSample(source_index);
  if(sample.is_object()) {
    return Segment::CopyDetectionSample(sample, target_index, data, metadata, class_manager, copy_mode);
  } else {
    return false;
  }
}

void SegmentSet::AddSegment(Segment *segment) {
  if(segment != nullptr) {
    segments_.push_back(segment);
    LOGDEBUG << "Added segment \"" << segment->name << "\" to set: \"" << name << "\"";
  } else {
    LOGERROR << "Tried to add null pointer segment to set: \"" << name << "\"";
  }
}

JSON SegmentSet::GetSample(unsigned int index) {
  auto segment_p = GetSegmentWithSampleIndex(index);
  Segment* const segment = segment_p.first;
  if(segment != nullptr) {
    const unsigned int in_segment_index = segment_p.second;
    return segment->GetSample(in_segment_index);
  } else {
    return false;
  }
}

unsigned int SegmentSet::GetSampleCount() const {
  unsigned int sample_count = 0;
  for (Segment *segment : segments_) {
    sample_count += segment->GetSampleCount();
  }
  return sample_count;
}


std::pair<Segment*, unsigned int> SegmentSet::GetSegmentWithSampleIndex(unsigned int index) {
  unsigned int in_segment_index = index;
  for (Segment *segment : segments_) {
    if (index < segment->GetSampleCount()) {
      std::pair<Segment *, unsigned int> p(segment, in_segment_index);
      return p;
    } else {
      index -= segment->GetSampleCount();
    }
  }
  std::pair<Segment *, unsigned int> p(nullptr, 0);
  return p;
}

JSON SegmentSet::Serialize() {
  JSON serialized_set = JSON::object();
  JSON serialized_segments = JSON::array();
  for(unsigned int s = 0; s < segments_.size(); s++) {
    JSON serialized_segment = segments_[s]->Serialize();
    serialized_segments.push_back(serialized_segment);
  }
  serialized_set["segments"] = serialized_segments;
  serialized_set["name"] = name;
  return serialized_set;
}

bool SegmentSet::Deserialize(JSON segment_set_descriptor, std::string folder_hint) {
  if(segment_set_descriptor.count("segments") == 1 && segment_set_descriptor["segments"].is_array()) {
    if(segment_set_descriptor.count("name") == 1 && segment_set_descriptor["name"].is_string()) {
      name = segment_set_descriptor["name"];
    }
    bool success = true;
    for(unsigned int s = 0; s < segment_set_descriptor["segments"].size(); s++) {
      std::string segment_name = "Unnamed segment";
      Segment* segment = new Segment(segment_name);
      success &= segment->Deserialize(segment_set_descriptor["segments"][s], folder_hint);
    }
    return success;
  } else {
    return false;
  }
}

}
