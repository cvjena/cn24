/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_SEGMENTSET_H
#define CONV_SEGMENTSET_H

#include "Segment.h"
#include "ClassManager.h"

#include <vector>
#include <utility>

namespace Conv {

class SegmentSet {
public:
  explicit SegmentSet(std::string name) : name(name) {}

  std::string name;

  bool CopyDetectionSample(
    unsigned int source_index,
    unsigned int target_index,
    Tensor* data,
    DetectionMetadataPointer metadata,
    ClassManager& class_manager,
    Segment::CopyMode copy_mode = Segment::NEVER_RESIZE);

  unsigned int GetSampleCount() const;
  JSON GetSample(unsigned int index);

  JSON Serialize();
  bool Deserialize(JSON segment_set_descriptor, std::string folder_hint = {});
  void AddSegment(Segment* segment);

  unsigned int GetSegmentCount() const { return segments_.size();}
  Segment* GetSegment(unsigned int index) { return segments_[index];}
  int GetSegmentIndex(const std::string& name) { for(unsigned int i = 0; i < segments_.size(); i++) if(segments_[i]->name.compare(name) == 0) return i; return -1; }
  bool RemoveSegment(unsigned int index) { if(index < segments_.size()) { segments_.erase(segments_.begin() + index); return true; } else { return false;}}

  bool RenameClass(const std::string& org_name, const std::string new_name);
private:
  std::pair<Segment*, unsigned int> GetSegmentWithSampleIndex(unsigned int index);
  std::vector<Segment*> segments_;
};
}
#endif
