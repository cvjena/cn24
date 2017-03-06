/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ShellState.h"

#include <string>
#include <fstream>

namespace Conv {

CN24_SHELL_FUNC_IMPL(ModelLoad) {
  CN24_SHELL_FUNC_DESCRIPTION("Loads a model from a CNParamX file");

  char* file = nullptr;
  char** skip_nodes = nullptr;
  std::size_t skip_nodes_count = 0;

  cargo_add_option(cargo, (cargo_option_flags_t)0, "file", "Model CNParamX file",
                   "s", &file);

  cargo_add_option(cargo, (cargo_option_flags_t)0, "--skip-nodes -s", "Skips the specified nodes while loading",
    "[s]+", &skip_nodes, &skip_nodes_count);

  CN24_SHELL_PARSE_ARGS;

  // Check if shell state allows for model loading
  if(state_ != NET_AND_TRAINER_LOADED && state_ != NET_LOADED) {
    LOGERROR << "Cannot load model, no network is loaded.";
    return FAILURE;
  }

  // Collect skip nodes
  std::vector<std::string> skip_nodes_str;
  for(int i = 0; i < skip_nodes_count; i++) {
    std::string skip_node_str = std::string(skip_nodes[i]);
    skip_nodes_str.push_back(skip_node_str);
  }

  // Find file
  std::string file_str(file);
  std::string path = PathFinder::FindPath(file_str, {});

  std::ifstream model_file(path, std::ios::in | std::ios::binary);
  if(path.length() == 0 || !model_file.good()) {
    LOGERROR << "Cannot open file: " << file;
    return FAILURE;
  }

  // Load model
  graph_->DeserializeParameters(model_file, skip_nodes_str);

  return SUCCESS;
}

CN24_SHELL_FUNC_IMPL(ModelSave) {
  CN24_SHELL_FUNC_DESCRIPTION("Saves a model to a CNParamX file");

  char* file = nullptr;
  char** skip_nodes = nullptr;
  std::size_t skip_nodes_count = 0;

  cargo_add_option(cargo, (cargo_option_flags_t)0, "file", "Model CNParamX file",
                   "s", &file);

  cargo_add_option(cargo, (cargo_option_flags_t)0, "--skip-nodes -s", "Skips the specified nodes while saving",
                   "[s]+", &skip_nodes, &skip_nodes_count);

  CN24_SHELL_PARSE_ARGS;

  // Check if shell state allows for model loading
  if(state_ != NET_AND_TRAINER_LOADED && state_ != NET_LOADED) {
    LOGERROR << "Cannot save model, no network is loaded.";
    return FAILURE;
  }

  // Collect skip nodes
  std::vector<std::string> skip_nodes_str;
  for(int i = 0; i < skip_nodes_count; i++) {
    std::string skip_node_str = std::string(skip_nodes[i]);
    skip_nodes_str.push_back(skip_node_str);
  }

  std::string file_str(file);

  std::ofstream model_file(file_str, std::ios::out | std::ios::binary);
  if(!model_file.good()) {
    LOGERROR << "Cannot open file: " << file;
    return FAILURE;
  }

  // Load model
  graph_->SerializeParameters(model_file, skip_nodes_str);

  return SUCCESS;
}
}
