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
    unsigned int class_id = it.value();
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
      for(std::map<std::string,unsigned int>::iterator mit = classes_.begin(); mit != classes_.end(); mit++) {
        if(mit->second == class_id) {
          // Same id, different name
          FATAL("Same class id " << class_id << " with different names " << class_name << " and " << mit->first << "!");
        }
      }

      // Id doesn't exist, okay
      classes_.emplace(class_name, class_id);
    }
  }
  return false;
}

JSON ClassManager::SaveToFile() {
  JSON dump = JSON::object();
  for(std::map<std::string,unsigned int>::iterator it = classes_.begin(); it != classes_.end(); it++) {
    dump[it->first] = it->second;
  }
  return dump;
}

bool ClassManager::RegisterClassByName(std::string name) {
  auto result = classes_.emplace(name, next_class_id_);

  if(result.second)
    next_class_id_++;

  return result.second;
}

unsigned int ClassManager::GetClassIdByName(const std::string& name) const {
  std::map<std::string,unsigned int>::const_iterator element = classes_.find(name);
  if(element != classes_.end())
    return element->second;
  return UNKNOWN_CLASS;
}

unsigned int ClassManager::GetMaxClassId() const {
  if(next_class_id_ == 0)
    return UNKNOWN_CLASS;
  return next_class_id_ - 1;
}

}
