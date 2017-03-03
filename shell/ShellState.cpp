/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ShellState.h"

#include <cstring>

namespace Conv {
  
ShellState::ShellState()
{
  Bundle* default_training_bundle = new Bundle("Default_Training");
  training_bundles_->push_back(default_training_bundle);
  training_weights_->push_back(1);
  Bundle* default_testing_bundle = new Bundle("Default_Testing");
  testing_bundles_->push_back(default_testing_bundle);
}

ShellState::CommandStatus ShellState::ProcessCommand(std::string command)
{
  // Skip empty commands
  if(command.length() == 0)
    return SUCCESS;
  
  // Skip comments
  if(command.length() > 0 && command[0] == '#')
    return SUCCESS;
  
  // Tokenize command list
  int cmd_argc = 1;
  
  // Remove double spaces and count spaces
  char* cmd_cstr = new char[command.length() + 1];
  for(int i = 0; i < command.length(); i++) cmd_cstr[i] = '\0';
  
  bool in_quote = false;
  char last_char = ' ';
  int cstr_pos = 0;
  for(int i = 0; i < command.length(); i++) {
    char current_char = command[i];
    bool skip = false;
    if(current_char == '\"') {
      in_quote = !in_quote;
    } else if(current_char == ' ' && !in_quote) {
      if(last_char != ' ') {
	skip = true;
	cmd_cstr[cstr_pos++] = '\0';
	last_char = ' ';
	cmd_argc++;
      } else {
	skip = true;
      }
    }
    
    if(!skip) {
      cmd_cstr[cstr_pos++] = current_char;
      last_char = current_char;
    }
  }
  cmd_cstr[cstr_pos] = '\0';
  
  // Split by zero bytes
  int cmd_cstr_len = cstr_pos;
  char** cmd_argv = new char*[cmd_argc];
  cmd_argv[0] = cmd_cstr;
  {
    int a = 1;
    for(int i = 0; i < cmd_cstr_len; i++) {
      if(cmd_cstr[i] == '\0') {
	cmd_argv[a++] = &(cmd_cstr[i+1]);
      }
    }
  }
  
  // Remove trailing stuff
  for(int a = (cmd_argc - 1); a > 0; a--) {
    if(cmd_argv[a][0] == '\0') {
      cmd_argc--;
    } else {
      break;
    }
  }
  
  // Find command
  const std::string command_name = std::string(cmd_argv[0]);
  std::map<std::string, ShellFunction>::iterator command_it = cmd_name_func_map.find(command_name);
  
  if(command_it == cmd_name_func_map.end()) {
    LOGERROR << "Unknown command: \"" << command_name << "\". Please enter"
    << " \"help\" for more information.";
    return WRONG_PARAMS;
  }
  
  // Initialize cargo
  cargo_t cargo;
  if(cargo_init(&cargo, (cargo_flags_t)(0), "%s", cmd_argv[0])) {
    LOGERROR << "Parser initialization failed.";
    return FAILURE;
  }
  
  ShellFunction function = command_it->second;
  CommandStatus result =(this->*function)(cargo, cmd_argc, cmd_argv, false);

  delete[] cmd_cstr;
  delete[] cmd_argv;
  return result;
}

CN24_SHELL_FUNC_IMPL(Quit) {
  CN24_SHELL_FUNC_DESCRIPTION("Exit CN24");
  CN24_SHELL_PARSE_ARGS;
  return REQUEST_QUIT;
}


CN24_SHELL_FUNC_IMPL(CommandHelp) {
  CN24_SHELL_FUNC_DESCRIPTION("Prints a list of commands");
  CN24_SHELL_PARSE_ARGS;
  for(std::map<std::string, ShellFunction>::const_iterator it = cmd_name_func_map.begin(); it != cmd_name_func_map.end(); it++) {
    const char* cmd = it->first.c_str();
    const char** argv;
    argv = &cmd; 
    cargo_t cargo_h;
    
    std::cout << it->first << ":" << std::endl << "  ";
    // Run the function with wrong parameters on purpose to print usage info
    cargo_init(&cargo_h, (cargo_flags_t)(CARGO_NO_AUTOHELP + CARGO_NOERR_OUTPUT), "%s", cmd);
    (this->*(it->second))(cargo_h, 1, (char**)argv, true);
  }
  return SUCCESS;
}

}