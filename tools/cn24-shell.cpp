/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>
#include <cstdlib>
#include <unistd.h>
#include <readline/readline.h>
#include <readline/history.h>

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
  
  // Readline loop
  char* shell_line = nullptr;
  while(true) {
    if(shell_line) {
      free(shell_line);
      shell_line = nullptr;
    }
    
    shell_line = readline ("\ncn24> ");
    
    if(shell_line && *shell_line) {
      add_history(shell_line);
    }
    
    // Process input
  }
  
  // Shutdown CN24
  Conv::System::Shutdown();
  LOGEND;
  return 0;

}
