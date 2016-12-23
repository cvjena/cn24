/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "PathFinder.h"
#include "Log.h"

#include <fstream>

#ifdef BUILD_POSIX
#include <unistd.h>
#include <sys/types.h>
#include <pwd.h>
#elif defined(BUILD_WIN32)
#include <Shlobj.h>
#endif

namespace Conv {

std::string PathFinder::FindPath(std::string path, std::string folder_hint) {
  std::string default_path = FindPathInternal(path, folder_hint);
  if(default_path.length() > 0)
    return default_path;

  std::string dotfolder_path = FindPathInternal(path, "~/.cn24");
  if(dotfolder_path.length() > 0)
    return dotfolder_path;

  std::string dotfolderwithhint_path = FindPathInternal(path, std::string("~/.cn24/") + folder_hint);
  if(dotfolderwithhint_path.length() > 0)
    return dotfolderwithhint_path;

  std::string winfolder_path = FindPathInternal(path, "~/cn24");
  if(winfolder_path.length() > 0)
    return winfolder_path;

  std::string winfolderwithhint_path = FindPathInternal(path, std::string("~/cn24/") + folder_hint);
  if(winfolderwithhint_path.length() > 0)
    return winfolderwithhint_path;

  return "";
}

std::string PathFinder::FindPathInternal(std::string path, std::string folder_hint) {

  // 1. See if path is valid on its own
  {
    std::ifstream path1_stream(path);
    if(path1_stream.good()) {
      return path;
    }
  }

#ifdef BUILD_POSIX
  // Find home directory
  const char* posix_home_dir = getenv("HOME");
  if(!posix_home_dir) {
    posix_home_dir = getpwuid(getuid())->pw_dir;
    if(!posix_home_dir) {
      posix_home_dir = "";
    }
  }

  std::string home_path = posix_home_dir;
#elif defined(BUILD_WIN32)
  std::string home_path = "";
  WCHAR windows_home_path[MAX_PATH];
  if (SUCCEEDED(SHGetFolderPathW(NULL, CSIDL_PROFILE, NULL, 0, windows_home_path))) {
    std::wstring windows_home_path_str = windows_home_path;
    home_path = std::string(windows_home_path_str.begin(), windows_home_path_str.end());
  }
#endif

  // 2. See if we can replace a tilde with the home folder
  {
    auto tilde_pos = path.find('~');
    if(tilde_pos == 0) {
      std::string path2 = path.replace(0, 1, home_path);
      std::ifstream path2_stream(path2);
      if(path2_stream.good()) {
        return path2;
      }
    }
  }

  // 3. See if folder hint works
  {
    std::string path3 = folder_hint + std::string("/") + path;
    std::ifstream path3_stream(path3);
    if(path3_stream.good()) {
      return path3;
    }
  }

  // 4. See if we can replace a tilde in the folder hint
  {
    auto tilde_pos = folder_hint.find('~');
    if (tilde_pos == 0) {
      std::string home_hint = folder_hint.replace(0, 1, home_path);
      std::string path4 = home_hint + "/" + path;
      std::ifstream path4_stream(path4);
      if (path4_stream.good()) {
        return path4;
      }
    }
  }

  // We have not found anything
  return "";
}
}
