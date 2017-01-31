/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */


#ifndef CONV_CLASSMANAGER_H
#define CONV_CLASSMANAGER_H

#include <string>
#include <map>

#include "Config.h"
#include "JSONParsing.h"

#define UNKNOWN_CLASS 999999

namespace Conv {
class ClassManager {
public:
  class ClassUpdateHandler {
  public:
    virtual void OnClassUpdate() = 0;
  };

  struct Info {
  public:
    unsigned int id = 0;
    unsigned int color = 0;
    datum weight = 0;
  };
  typedef std::map<std::string,Info>::const_iterator const_iterator;

  ClassManager();

  // Event handling
  void RegisterClassUpdateHandler(ClassUpdateHandler* handler) { handlers_.push_back(handler); }

  // Iterate
  const_iterator begin() { return classes_.begin(); }
  const_iterator end() { return classes_.end(); }

  // Import/Export
  bool LoadFromFile(JSON configuration);
  JSON SaveToFile();

  bool RegisterClassByName(std::string name, unsigned int color, datum weight);
  unsigned int GetClassIdByName(const std::string& name) const;
  Info GetClassInfoByName(const std::string& name) const;
  std::pair<std::string, Info> GetClassInfoById(unsigned int id) const;

  unsigned int GetMaxClassId() const;
  unsigned int GetClassCount() const { return classes_.size(); }

  bool RenameClass(const std::string& org_name, const std::string new_name);
private:
  std::map<std::string,Info> classes_;
  std::map<unsigned int, std::pair<std::string,Info>> by_id_;
  std::vector<ClassUpdateHandler*> handlers_;
  unsigned int next_class_id_ = 0;
};
}

#endif
