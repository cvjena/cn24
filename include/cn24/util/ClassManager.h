/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */


#ifndef CONV_CLASSMANAGER_H
#define CONV_CLASSMANAGER_H

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
  unsigned int GetClassIdByName(std::string name);

  unsigned int GetMaxClassId();
};
}

#endif
