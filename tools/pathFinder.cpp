/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>

int main(int argc, char** argv) {
  Conv::System::Init(3);

  while(true) {
    std::cout << "\n file > ";

    // Read user input
    std::string path, hint;
    std::getline(std::cin, path);

    std::cout << " hint > ";
    std::getline(std::cin, hint);

    std::string found_path = Conv::PathFinder::FindPath(path, hint);
    LOGINFO << "Found path: " << found_path;
  }


  LOGEND;
}
