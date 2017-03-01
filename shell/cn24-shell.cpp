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

#include "ShellState.h"

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
  
  // Initialize shell state object
  Conv::ShellState shell_state;
  
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
    std::string shell_line_str(shell_line);
    if(shell_line_str.compare("q") == 0 ||
      shell_line_str.compare("quit") == 0 ||
      shell_line_str.compare("exit") == 0
    ) {
      break;
    } else {
      shell_state.ProcessCommand(shell_line_str);
    }
  }
  
  // Shutdown CN24
  Conv::System::Shutdown();
  LOGEND;
  return 0;

}
