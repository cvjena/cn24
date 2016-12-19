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
  return JSON::object();
}

bool SegmentSet::Deserialize(JSON segment_set_descriptor) {
  return false;
}

}
