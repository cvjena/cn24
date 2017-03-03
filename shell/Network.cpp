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
  int predict_only = -1;
  int seed = global_random_seed;
  cargo_add_option(cargo, (cargo_option_flags_t)0, "file", "JSON file"
  " containing the architecture", "s", &file);
  cargo_add_option(cargo, (cargo_option_flags_t)CARGO_OPT_NOT_REQUIRED, "task", "Task the network"
  " is going to be used for", "s", &task);
  cargo_add_validation(cargo, (cargo_validation_flags_t)0, "task",
    cargo_validate_choices((cargo_validate_choices_flags_t)
    CARGO_VALIDATE_CHOICES_CASE_SENSITIVE, CARGO_STRING,
    3, "detection", "classification", "segmentation"));
  cargo_add_option(cargo, (cargo_option_flags_t)0, "--seed", "Random seed"
  " for weight initialization", "i", &seed);
  cargo_add_validation(cargo, (cargo_validation_flags_t)0, "--seed",
    cargo_validate_int_range(0, 99999999));
  cargo_add_option(cargo, (cargo_option_flags_t)0, "--predict-only",
    "Skips Trainer initialization. Saves some memory, but disables training",
    "b", &predict_only);
  
  CN24_SHELL_PARSE_ARGS;
  
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
  
  Task task_ = UNKNOWN;
  if(task != nullptr) {
    std::string task_str(task);
    task_ = TaskFromString(task_str);
  } else {
    task_ = factory.GetTask();
  }
  
  if(task_ == UNKNOWN) {
    LOGERROR << "Could not determine task from either command line or architecture";
    delete class_manager_;
    class_manager_ = nullptr;
    return WRONG_PARAMS;
  }
  
  // Create input layer
  unsigned int batch_size = factory.GetParallelBatchSize();
  input_layer_ = new BundleInputLayer(factory.GetDataInput(), task_,
    class_manager_, batch_size, (unsigned int)(seed + 1));
  
  NetGraphNode* input_node = new NetGraphNode(input_layer_);
  input_node->is_input = true;
  input_node->unique_name = "segmentsetinput";
  
  // Assemble network
  graph_ = new NetGraph();
  graph_->AddNode(input_node);
  
  bool add_result =
    factory.AddLayers(*graph_, class_manager_, (unsigned int)(seed + 2));
  
  if(!add_result) {
    LOGERROR << "Could not add layers to net graph! Destroying net.";
    delete graph_; graph_ = nullptr;
    delete input_layer_; input_layer_ = nullptr;
    delete class_manager_; class_manager_ = nullptr;
    return FAILURE;
  }
  
  // Initialize DAG
  graph_->Initialize();
  graph_->InitializeWeights(true);
  
  state_ = NET_LOADED;
  
  if(predict_only == 1)
    return SUCCESS;
  
  // Create trainer
  trainer_ = new Trainer(*graph_, factory.GetHyperparameters());
  state_ = NET_AND_TRAINER_LOADED;
  
  return SUCCESS;
}

CN24_SHELL_FUNC_IMPL(NetworkStatus) {
  CN24_SHELL_FUNC_DESCRIPTION("Displays information about the current network") 
  CN24_SHELL_PARSE_ARGS;
  
  switch(state_) {
    case NOTHING:
      LOGINFO << "No network loaded.";
      break;
    case NET_LOADED:
      LOGINFO << "Network loaded for prediction only.";
      break;
    case NET_AND_TRAINER_LOADED:
      LOGINFO << "Network loaded.";
      break;
  }
  switch(state_) {
    case NOTHING:
      break;
    case NET_AND_TRAINER_LOADED:
      LOGINFO << "Current epoch: " << trainer_->epoch();
    case NET_LOADED:
      LOGINFO << "Network nodes: " << graph_->GetNodes().size();
      break;
  }
  return SUCCESS;
}

}