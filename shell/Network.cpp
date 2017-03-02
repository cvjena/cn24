/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ShellState.h"

#include <fstream>
#include <exception>

namespace Conv {
  
CN24_SHELL_FUNC_IMPL(NetworkLoad) {
  CN24_SHELL_FUNC_DESCRIPTION("Loads a network architecture");
  
  char* file = nullptr;
  char* task = nullptr;
  int seed = 0;
  cargo_add_option(cargo, (cargo_option_flags_t)0, "file", "JSON file"
  " containing the architecture", "s", &file);
  cargo_add_option(cargo, (cargo_option_flags_t)0, "task", "Task the network"
  " is going to be used for", "s", &task);
  cargo_add_validation(cargo, (cargo_validation_flags_t)0, "task",
    cargo_validate_choices((cargo_validate_choices_flags_t)
    CARGO_VALIDATE_CHOICES_CASE_SENSITIVE, CARGO_STRING,
    3, "detection", "classification", "segmentation"));
  cargo_add_option(cargo, (cargo_option_flags_t)0, "--seed", "Random seed"
  " for weight initialization", "i", &seed);
  cargo_add_validation(cargo, (cargo_validation_flags_t)0, "--seed",
    cargo_validate_int_range(0, 99999999));
  
  CN24_SHELL_PARSE_ARGS;
  
  Task task_ = CLASSIFICATION;
  std::string task_str(task);
  if(task_str.compare("detection") == 0) {
    task_ = DETECTION;
  } else if(task_str.compare("classification") == 0) {
    task_ = CLASSIFICATION;
  } else if(task_str.compare("segmentation") == 0) {
    task_ = SEMANTIC_SEGMENTATION;
  } else {
    return WRONG_PARAMS;
  }
  
  if(state_ != NOTHING) {
    LOGERROR << "Network already loaded!";
    return FAILURE;
  }
  
  // Load architecture
  std::string path = PathFinder::FindPath(std::string(file), {});
  std::ifstream arch_file(path, std::ios::in);
  if(path.length() == 0 || !arch_file.good()) {
    LOGERROR << "Cannot open file: " << file;
    return FAILURE;
  }
  
  // Parse architecture
  JSON architecture_json;
  try {
        architecture_json = JSON::parse(arch_file);
  } catch(std::exception ex) {
    LOGERROR << "Could not correctly parse network architecture:";
    LOGERROR << ex.what();
    return FAILURE;
  }
  
  // Create class manager
  class_manager_ = new ClassManager();
  // Use factory
  JSONNetGraphFactory factory = JSONNetGraphFactory(architecture_json,
    (unsigned int)seed);
  
  // Create input layer
  unsigned int batch_size = factory.GetParallelBatchSize();
  input_layer_ = new SegmentSetInputLayer(factory.GetDataInput(), task_,
    class_manager_, batch_size, (unsigned int)(seed + 1));
  
  NetGraphNode* input_node = new NetGraphNode(input_layer_);
  input_node->is_input = true;
  input_node->unique_name = "segmentsetinput";
  
  // Assemble network
  graph_ = new NetGraph();
  graph_->AddNode(input_node);
  
  factory.AddLayers(*graph_, class_manager_, (unsigned int)(seed + 2));
  
  return SUCCESS;
}

}