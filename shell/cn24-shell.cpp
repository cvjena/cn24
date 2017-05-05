/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>
#include <cstdlib>
#include <iostream>

#include "linenoise.h"
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
  char* cmdl_script = nullptr;

  int success = 0;
  success |= cargo_add_option(cargo, (cargo_option_flags_t)0, "--verbose -v",
    "Verbose mode (for debugging)", "b", &cmdl_verbose);
  success |= cargo_add_option(cargo, (cargo_option_flags_t)0, "--quiet -q",
    "Quiet mode (for scripting)", "b", &cmdl_quiet);
  success |= cargo_add_option(cargo, (cargo_option_flags_t)CARGO_OPT_NOT_REQUIRED, "script",
    "Script to run instead of command line", "s", &cmdl_script);

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

  // Initialize linenoise
  linenoiseInstallWindowChangeHandler();

  // Initialize CN24
  Conv::System::Init(cmdl_log_level);
  
  // Initialize shell state object
  Conv::ShellState shell_state;

  if(cmdl_script != nullptr && cmdl_script[0] != '\0') {
    // Run script
    std::string script_file = cmdl_script;
    Conv::ShellState::CommandStatus status = shell_state.ProcessScript(script_file, false);
    switch(status) {
      case Conv::ShellState::SUCCESS:
      case Conv::ShellState::REQUEST_QUIT:
        break;
      case Conv::ShellState::WRONG_PARAMS:
      case Conv::ShellState::FAILURE:
        LOGERROR << "Excecution of " << script_file << " aborted.";
        break;
    }
  } else {
    shell_state.OfferCommandLine("cn24>");
  }
  
  // Shutdown CN24
  Conv::System::Shutdown();
  LOGEND;
  return 0;

}
