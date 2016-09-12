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
  return false;
}

JSON ClassManager::SaveToFile() {
  return JSON::object();
}

bool ClassManager::RegisterClassByName(std::string name) {
  return false;
}

unsigned int ClassManager::GetClassIdByName(std::string name) {
  return UNKNOWN_CLASS;
}

unsigned int ClassManager::GetMaxClassId() {
  return 0;
}

}
