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
  typedef std::map<std::string,unsigned int>::const_iterator const_iterator;

  ClassManager();

  // Iterate
  const_iterator begin() { return classes_.begin(); }
  const_iterator end() { return classes_.end(); }

  // Import/Export
  bool LoadFromFile(JSON configuration);
  JSON SaveToFile();

  bool RegisterClassByName(std::string name);
  unsigned int GetClassIdByName(const std::string& name) const;

  unsigned int GetMaxClassId() const;
  unsigned int GetClassCount() const { return classes_.size(); }

private:
  std::map<std::string,unsigned int> classes_;
  unsigned int next_class_id_ = 0;
};
}

#endif
