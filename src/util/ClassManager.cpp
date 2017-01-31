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

bool ClassManager::RenameClass(const std::string &org_name, const std::string new_name) {
  Info class_info = GetClassInfoByName(org_name);
  if(class_info.id == UNKNOWN_CLASS) {
    LOGERROR << "Class \"" << org_name << "\" not found!";
    return false;
  } else {
    // Class found
    // Update classes_ map
    classes_.erase(org_name);
    classes_.emplace(new_name, class_info);

    // Update by_id map
    std::pair<std::string,Info> p;
    p.first = new_name;
    p.second = class_info;
    by_id_.erase(class_info.id);
    by_id_.emplace(class_info.id, p);

    return true;
  }
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
      class_info.weight = obj["weight"];
      class_info.color = obj["color"];
      classes_.emplace(class_name, class_info);
      std::pair<std::string,Info> p;
      p.first = class_name;
      p.second = class_info;
      by_id_.emplace(class_info.id, p);
    }
  }
  for(std::vector<ClassUpdateHandler*>::iterator it = handlers_.begin(); it != handlers_.end(); it++)(*it)->OnClassUpdate();
  return true;
}

JSON ClassManager::SaveToFile() {
  JSON dump = JSON::object();
  for(std::map<std::string,Info>::iterator it = classes_.begin(); it != classes_.end(); it++) {
    JSON obj = JSON::object();
    obj["id"] = it->second.id;
    obj["color"] = it->second.color;
    obj["weight"] = it->second.weight;
    dump[it->first] = obj;
  }
  return dump;
}

bool ClassManager::RegisterClassByName(std::string name, unsigned int color, datum weight) {
  if(GetClassIdByName(name) != UNKNOWN_CLASS) {
    Info current_info = GetClassInfoByName(name);
    if(current_info.color != color) {
      LOGWARN << "Class \"" << name << "\" color mismatch";
    }
    if(current_info.weight != weight) {
      LOGWARN << "Class \"" << name << "\" weight mismatch";
    }
    return true;
  } else {
    Info info;
    info.id = next_class_id_;
    info.color = color;
    info.weight = weight;
    auto result = classes_.emplace(name, info);

    if (result.second) {
      std::pair<std::string, Info> p;
      p.first = name;
      p.second = info;
      by_id_.emplace(info.id, p);
      next_class_id_++;
    }

    for (std::vector<ClassUpdateHandler *>::iterator it = handlers_.begin(); it != handlers_.end(); it++)
      (*it)->OnClassUpdate();
    return result.second;
  }
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

std::pair<std::string, ClassManager::Info> ClassManager::GetClassInfoById(unsigned int id) const {
  std::map<unsigned int, std::pair<std::string,Info>>::const_iterator element = by_id_.find(id);
  if(element != by_id_.end()) {
    return element->second;
  }
  std::pair<std::string, Info> p;
  p.first = "Unknown";
  p.second.id = UNKNOWN_CLASS;
  return p;
}

unsigned int ClassManager::GetMaxClassId() const {
  if(next_class_id_ == 0)
    return 0;
  return next_class_id_ - 1;
}

}
