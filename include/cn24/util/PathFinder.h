/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#ifndef CONV_PATHFINDER_H
#define CONV_PATHFINDER_H

#include <string>

namespace Conv {

class PathFinder {
public:
  static std::string FindPath(std::string path, std::string folder_hint);

private:
  static std::string FindPathInternal(std::string path, std::string folder_hint);
};

}

#endif
