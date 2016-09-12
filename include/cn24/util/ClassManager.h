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
  ClassManager();

  // Import/Export
  bool LoadFromFile(JSON configuration);
  JSON SaveToFile();

  bool RegisterClassByName(std::string name);
  unsigned int GetClassIdByName(const std::string& name) const;

  unsigned int GetMaxClassId() const;

private:
  std::map<std::string,unsigned int> classes_;
  unsigned int next_class_id_ = 0;
};
}

#endif
