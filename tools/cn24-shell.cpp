/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>
#include <unistd.h>

int main(int argc, char** argv) {
  int cmdl_log_level = -1;
  // Parse command line options
  char c;
  while((c = getopt(argc, argv, "vq")) != -1) {
    switch(c) {
      case 'v':
	cmdl_log_level = 3;
	break;
      case 'q':
	cmdl_log_level = 0;
	break;
      default:
	break;
    }
  }
  
  // Initialize CN24
  Conv::System::Init(cmdl_log_level);
  
  // Shutdown CN24
  Conv::System::Shutdown();
  LOGEND;
  return 0;

}
