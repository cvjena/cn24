/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Log.h"

#include "ClassManager.h"

namespace Conv {

ClassManager::ClassManager() {
  LOGDEBUG << "Instance created.";
}

bool ClassManager::LoadFromFile(JSON configuration) {
  for(JSON::iterator it = configuration.begin(); it != configuration.end(); it++) {
    // Check if class name exists
    std::string class_name = it.key();
    JSON obj = it.value();
    unsigned int class_id = obj["id"];
    unsigned int registered_id = GetClassIdByName(class_name);
    if(registered_id != UNKNOWN_CLASS) {
      // Class name exists
      if(registered_id == class_id) {
        // OK, skip this entry
        continue;
      } else {
        // Throw error, same class name with different ids
        FATAL("Same class name " << class_name << " with different ids " << class_id << " and " << registered_id << "!");
      }
    } else {
      // Class is new, check if id exists
      for(std::map<std::string,Info>::iterator mit = classes_.begin(); mit != classes_.end(); mit++) {
        if(mit->second.id == class_id) {
          // Same id, different name
          FATAL("Same class id " << class_id << " with different names " << class_name << " and " << mit->first << "!");
        }
      }

      // Id doesn't exist, okay
      Info class_info; class_info.id = class_id;
      classes_.emplace(class_name, class_info);
    }
  }
  return false;
}

JSON ClassManager::SaveToFile() {
  JSON dump = JSON::object();
  for(std::map<std::string,Info>::iterator it = classes_.begin(); it != classes_.end(); it++) {
    JSON obj = JSON::object();
    obj["id"] = it->second.id;
    dump[it->first] = obj;
  }
  return dump;
}

bool ClassManager::RegisterClassByName(std::string name, unsigned int color, datum weight) {
  Info info;
  info.id = next_class_id_;
  info.color = color;
  info.weight = weight;
  auto result = classes_.emplace(name, info);

  if(result.second)
    next_class_id_++;

  return result.second;
}

unsigned int ClassManager::GetClassIdByName(const std::string& name) const {
  std::map<std::string,Info>::const_iterator element = classes_.find(name);
  if(element != classes_.end())
    return element->second.id;
  return UNKNOWN_CLASS;
}

ClassManager::Info ClassManager::GetClassInfoByName(const std::string& name) const {
  std::map<std::string,Info>::const_iterator element = classes_.find(name);
  if(element != classes_.end())
    return element->second;

  Info info; info.id = UNKNOWN_CLASS;
  return info;
}

unsigned int ClassManager::GetMaxClassId() const {
  if(next_class_id_ == 0)
    return UNKNOWN_CLASS;
  return next_class_id_ - 1;
}

}
