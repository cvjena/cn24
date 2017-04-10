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

bool Segment::CopyDetectionSample(JSON& sample, unsigned int target_index, Tensor *data,
                                  DetectionMetadataPointer metadata, ClassManager& class_manager,
                                  CopyMode copy_mode) {
  // Load image data
  Tensor image_rgb;
  image_rgb.LoadFromFile(sample["image_rpath"]);

  // Copy data
  bool data_success = Tensor::CopySample(image_rgb, 0, *data, target_index, copy_mode != NEVER_RESIZE, copy_mode == SCALE);
  if(!data_success) {
    LOGERROR << "Could not copy sample for " << sample["image_rpath"];
    LOGERROR << "Tensor proportions: " << image_rgb;
  }

  bool metadata_success = true;

  // Copy metadata
  if(copy_mode != CROP) {
    metadata_success = CopyDetectionMetadata(sample, image_rgb.width(), image_rgb.height(), class_manager, metadata);
  } else {
    LOGERROR << "Cropping for detection is not implemented yet!";
  }

  return data_success && metadata_success;
}

bool Segment::CopyDetectionMetadata(JSON& sample, unsigned int image_width, unsigned int image_height, ClassManager &class_manager, DetectionMetadataPointer metadata) {
  metadata->clear();

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
        const datum width = image_width;
        const datum height = image_height;
        box.x /= width;
        box.w /= width;
        box.y /= height;
        box.h /= height;
      }

      metadata->push_back(box);
    }
  } else {
    LOGERROR << "Sample is missing metadata: " << sample.dump();
    return false;
  }
  return true;
}

bool Segment::CopyClassificationSample(JSON& sample, unsigned int target_index,
  Tensor* data, Tensor* label, ClassManager& class_manager, Segment::CopyMode copy_mode)
{
  // Load image data
  Tensor image_rgb;
  image_rgb.LoadFromFile(sample["image_rpath"]);

  // Copy data
  bool data_success = Tensor::CopySample(image_rgb, 0, *data, target_index, copy_mode != NEVER_RESIZE, copy_mode == SCALE);
  if(!data_success) {
    LOGERROR << "Could not copy sample for " << sample["image_rpath"];
    LOGERROR << "Tensor proportions: " << image_rgb;
  }

  bool label_success = true;
  
  // Copy label
  if(sample.count("class") == 1 && sample["class"].is_string()) {
    const std::string& class_name = sample["class"];
    int class_id = class_manager.GetClassIdByName(class_name);
    if(class_id == UNKNOWN_CLASS) {
      LOGDEBUG << "Autoregistering class " << class_name;
      if(!class_manager.RegisterClassByName(class_name, 0, 1)) {
        LOGERROR << "Could not register class \"" << class_name << "\"";
        return false;
      }
    }
    
    class_id = class_manager.GetClassIdByName(class_name);
    
    // See if tensor dimension are okay
    if(label->maps() <= class_id) {
      label->Resize(label->samples(), label->width(), label->height(), class_id + 1);
      LOGDEBUG << "Extended label tensor";
    }
    
    // Write label
    label->Clear(0,target_index);
    *(label->data_ptr(0,0,class_id,target_index)) = 1.0;
  } else {
    label_success = false;
  }

  return data_success && label_success;
}

bool Segment::CopyBinarySegmentationSample(JSON &sample, unsigned int target_index, Tensor *data, Tensor *label,
                                           ClassManager &class_manager, CopyMode copy_mode) {
  // Load image data
  Tensor image_rgb;
  image_rgb.LoadFromFile(sample["image_rpath"]);

  // Copy data
  bool data_success = Tensor::CopySample(image_rgb, 0, *data, target_index, copy_mode != NEVER_RESIZE, copy_mode == SCALE);
  if(!data_success) {
    LOGERROR << "Could not copy sample for " << sample["image_rpath"];
    LOGERROR << "Tensor proportions: " << image_rgb;
  }

  // Load label data
  label->Clear(0, target_index);

  Tensor label_rgb;
  if(sample.count("label_rpath") == 1 && sample["label_rpath"].is_string()) {
    label_rgb.LoadFromFile(sample["label_rpath"]);

    // Copy label
    bool label_success = Tensor::CopyMap(label_rgb, 0, 0, *label, target_index, 0, copy_mode != NEVER_RESIZE,
                                         copy_mode == SCALE);
    if (!label_success) {
      LOGERROR << "Could not copy sample for " << sample["image_rpath"];
      LOGERROR << "Tensor proportions: " << image_rgb;
    }

    return data_success && label_success;
  } else {
    return data_success;
  }
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
  serialized["folder_hint"] = last_folder_hint_;
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
    return true;
  } else {
    // Resolve relative paths

    // (1) Segmentation labels
    if (sample_descriptor.count("label_filename") == 1 && sample_descriptor["label_filename"].is_string()) {
      std::string label_filename = sample_descriptor["label_filename"];
      std::string resolved_path = PathFinder::FindPath(label_filename, folder_hint);

      if (resolved_path.length() > 0 && folder_hint.length() > 0)
        last_folder_hint_ = folder_hint;

      if (resolved_path.length() == 0)
        resolved_path = PathFinder::FindPath(label_filename, last_folder_hint_);

      if (resolved_path.length() > 0) {
        sample_descriptor["label_rpath"] = resolved_path;
      } else {
        LOGERROR << "Could not find sample label \"" << label_filename << "\", skipping!";
        return false;
      }
    }

    // (2) Image files
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

bool Segment::RenameClass(const std::string &org_name, const std::string new_name) {
  for(unsigned int s = 0; s < samples_.size(); s++) {
    Conv::JSON& sample_json = samples_[s];
    if(sample_json.count("boxes") == 1 && sample_json["boxes"].is_array()) {
      // Detection sample...
      Conv::JSON& boxes_json = sample_json["boxes"];
      for(unsigned int b = 0; b < boxes_json.size(); b++) {
        Conv::JSON& box_json = boxes_json[b];
        if(box_json.count("class") == 1 && box_json["class"].is_string()) {
          std::string original_class = box_json["class"];
          if(original_class.compare(org_name) == 0) {
            box_json["class"] = new_name;
          }
        } else {
          LOGERROR << "Sample has box without class: " << sample_json.dump();
          return false;
        }
      }
    }
    else if(sample_json.count("image_class") == 1 && sample_json["image_class"].is_string()) {
      // Classification sample...
    } else {
      // Don't know? Warn the user.
      LOGERROR << "Sample has no class information! " << sample_json.dump();
      return false;
    }
  }
  return true;
}
}
