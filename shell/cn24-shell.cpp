/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>
#include <cstdlib>
#include <iostream>
#include <readline/readline.h>
#include <readline/history.h>

#include "ShellState.h"
extern "C" {
  #include "cargo.h"
}

int main(int argc, char** argv) {
  
  // Parse command line options
  cargo_t cargo;
  if(cargo_init(&cargo, (cargo_flags_t)0, "%s", argv[0])) {
    std::cerr << "Failed to initialize command line parser!" << std::endl;
    return -1;
  }
  
  int cmdl_log_level = -1;
  
  // Flags
  int cmdl_verbose = 0;
  int cmdl_quiet = 0;
  char* cmdl_network = nullptr;
  
  int success = 0;
  success |= cargo_add_option(cargo, (cargo_option_flags_t)0, "--verbose -v",
    "Verbose mode (for debugging)", "b", &cmdl_verbose);
  success |= cargo_add_option(cargo, (cargo_option_flags_t)0, "--quiet -q",
    "Quiet mode (for scripting)", "b", &cmdl_quiet);
  success |= cargo_add_option(cargo, (cargo_option_flags_t)0, "--net",
    "Specify network architecture", "s", &cmdl_network);
  
  if(success != 0) {
    std::cerr << "Failed to initialize command line parser!" << std::endl;
    return -1;
  }
  
  // Run parser
  if(cargo_parse(cargo, (cargo_flags_t)0, 1, argc, argv)) {
    return -1;
  }
  
  // Process parsing result
  if(cmdl_verbose == 1) {
    cmdl_log_level = 3;
  } else if(cmdl_quiet == 1) {
    cmdl_log_level = 0;
  }
    
  // Initialize CN24
  Conv::System::Init(cmdl_log_level);
  
  // Initialize shell state object
  Conv::ShellState shell_state;
  
  // Readline loop
  char* shell_line = nullptr;
  bool process_commands = true;
  while(process_commands) {
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

    Conv::ShellState::CommandStatus status = shell_state.ProcessCommand(shell_line_str);
    switch(status) {
      case Conv::ShellState::SUCCESS:
      case Conv::ShellState::WRONG_PARAMS:
	break;
      case Conv::ShellState::FAILURE:
	LOGERROR << "Command execution failed.";
	break;
      case Conv::ShellState::REQUEST_QUIT:
	process_commands = false;
	break;
    }
  }
  
  // Shutdown CN24
  Conv::System::Shutdown();
  LOGEND;
  return 0;

}
