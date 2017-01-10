/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Segment.h"

#include "PathFinder.h"
#include "Tensor.h"
#include "Log.h"

namespace Conv {

bool Segment::CopyDetectionSample(JSON sample, unsigned int target_index, Tensor *data,
                                  DetectionMetadataPointer metadata, ClassManager& class_manager,
                                  CopyMode copy_mode) {
  // Load image data
  Tensor image_rgb;
  image_rgb.LoadFromFile(sample["image_rpath"]);

  // Copy data
  bool data_success = Tensor::CopySample(image_rgb, 0, *data, target_index, copy_mode != NEVER_RESIZE, copy_mode == SCALE);

  bool metadata_success = true;
  metadata->clear();

  // Copy metadata
  if(copy_mode != CROP) {
    if(sample.count("boxes") == 1 && sample["boxes"].is_array()) {
      for(unsigned int b = 0; b < sample["boxes"].size(); b++) {
        JSON& box_json = sample["boxes"][b];
        BoundingBox box(box_json["x"], box_json["y"], box_json["w"], box_json["h"]);
        if(box_json.count("difficult") == 1 && box_json["difficult"].is_number()) {
          unsigned int difficult = box_json["difficult"];
          box.flag2 = difficult > 0;
        }

        // Find the class by name
        std::string class_name = box_json["class"];
        box.c = class_manager.GetClassIdByName(class_name);
        bool class_found = box.c != UNKNOWN_CLASS;

        if(!class_found) {
          LOGDEBUG << "Autoregistering class " << class_name;
          class_manager.RegisterClassByName(class_name, 0, 1.0);
          box.c = class_manager.GetClassIdByName(class_name);
        }

        // Scale the box coordinates
        bool dont_scale = false;
        JSON_TRY_BOOL(dont_scale, box_json, "dont_scale", false);

        if(!dont_scale) {
          const datum width = image_rgb.width();
          const datum height = image_rgb.height();
          box.x /= width;
          box.w /= width;
          box.y /= height;
          box.h /= height;
        }

        metadata->push_back(box);
      }
    } else {
      LOGERROR << "Sample is missing metadata: " << sample.dump();
      metadata_success = false;
    }
  } else {
    LOGERROR << "Cropping for detection is not implemented yet!";
  }

  return data_success && metadata_success;
}

JSON Segment::Serialize() {
  JSON serialized = JSON::object();
  JSON samples_array = JSON::array();
  for(unsigned int s = 0; s < samples_.size(); s++) {
    JSON sample = samples_[s];
    sample.erase("image_rpath");
    samples_array.push_back(samples_[s]);
  }
  serialized["samples"] = samples_array;
  serialized["name"] = name;
  serialized["last_folder_hint"] = last_folder_hint_;
  return serialized;
}

bool Segment::Deserialize(JSON segment_descriptor, std::string folder_hint, int range_begin, int range_end) {
  bool success = true;

  if(segment_descriptor.count("folder_hint") == 1 && segment_descriptor["folder_hint"].is_string()) {
    last_folder_hint_ = segment_descriptor["folder_hint"];
  }

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
  if(segment_descriptor.count("name") == 1 && segment_descriptor["name"].is_string()) {
    name = segment_descriptor["name"];
  }
  return success;
}

bool Segment::AddSample(JSON sample_descriptor, std::string folder_hint, bool use_rpath) {
  if(use_rpath) {
    samples_.push_back(sample_descriptor);
  } else {
    if (sample_descriptor.count("image_filename") == 1 && sample_descriptor["image_filename"].is_string()) {
      std::string image_filename = sample_descriptor["image_filename"];
      std::string resolved_path = PathFinder::FindPath(image_filename, folder_hint);

      if (resolved_path.length() > 0 && folder_hint.length() > 0)
        last_folder_hint_ = folder_hint;

      if (resolved_path.length() == 0)
        resolved_path = PathFinder::FindPath(image_filename, last_folder_hint_);

      if (resolved_path.length() > 0) {
        sample_descriptor["image_rpath"] = resolved_path;
        samples_.push_back(sample_descriptor);
        return true;
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
}
