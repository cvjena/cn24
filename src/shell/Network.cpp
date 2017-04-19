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

  if(predict_only != 1) {
    // Add stat layers
    if (task_ == Conv::SEMANTIC_SEGMENTATION || task_ == Conv::CLASSIFICATION) {
      for (Conv::NetGraphNode *output_node : graph_->GetOutputNodes()) {
        // Add appropriate statistics layer
        Conv::NetGraphNode *stat_node = nullptr;
        if (class_manager_->GetClassCount() == 1) {
          Conv::BinaryStatLayer *binary_stat_layer = new Conv::BinaryStatLayer(13, -1, 1);
          stat_node = new Conv::NetGraphNode(binary_stat_layer);
        } else {
          Conv::ConfusionMatrixLayer *confusion_matrix_layer = new Conv::ConfusionMatrixLayer(class_manager_);
          stat_node = new Conv::NetGraphNode(confusion_matrix_layer);
        }
        stat_node->input_connections.push_back(Conv::NetGraphConnection(output_node, 0, false));
        stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 1));
        stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 3));
        graph_->AddNode(stat_node);
      }
    } else if (task_ == Conv::BINARY_SEGMENTATION) {
      for (Conv::NetGraphNode *output_node : graph_->GetOutputNodes()) {
        // Add appropriate statistics layer
        Conv::NetGraphNode *stat_node = nullptr;
        Conv::BinaryStatLayer *binary_stat_layer = new Conv::BinaryStatLayer(13, -1, 1);
        stat_node = new Conv::NetGraphNode(binary_stat_layer);

        stat_node->input_connections.push_back(Conv::NetGraphConnection(output_node, 0, false));
        stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 1));
        stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 3));

        graph_->AddNode(stat_node);
      }
    } else if (task_ == Conv::DETECTION) {
      for (Conv::NetGraphNode *output_node : graph_->GetOutputNodes()) {
        // Add appropriate statistics layer
        Conv::NetGraphNode *stat_node = nullptr;
        Conv::DetectionStatLayer *detection_stat_layer = new Conv::DetectionStatLayer(class_manager_);

        stat_node = new Conv::NetGraphNode(detection_stat_layer);
        stat_node->input_connections.push_back(Conv::NetGraphConnection(output_node, 0, false));
        stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 1));
        stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 3));
        graph_->AddNode(stat_node);
      }
    } else {
      LOGWARN << "I don\'t know what statistics layers to add for this task!";
    }
  }
  
  // Initialize DAG
  graph_->Initialize();
  graph_->InitializeWeights(true);
  
  state_ = NET_LOADED;
  
  // Switch data around
  for(unsigned int i = 0; i < training_bundles_->size(); i++)
    input_layer_->training_sets_.push_back(training_bundles_->at(i));
  for(unsigned int i = 0; i < training_weights_->size(); i++)
    input_layer_->training_weights_.push_back(training_weights_->at(i));
  for(unsigned int i = 0; i < staging_bundles_->size(); i++)
    input_layer_->staging_sets_.push_back(staging_bundles_->at(i));
  for(unsigned int i = 0; i < testing_bundles_->size(); i++)
    input_layer_->testing_sets_.push_back(testing_bundles_->at(i));
  delete training_bundles_; delete training_weights_;
  delete staging_bundles_;
  delete testing_bundles_;
  training_bundles_ = &(input_layer_->training_sets_);
  training_weights_ = &(input_layer_->training_weights_);
  staging_bundles_ = &(input_layer_->staging_sets_);
  testing_bundles_ = &(input_layer_->testing_sets_);
  
  if(predict_only == 1)
    return SUCCESS;
  
  // Create trainer
  trainer_ = new Trainer(*graph_, factory.GetHyperparameters());
  state_ = NET_AND_TRAINER_LOADED;

  // Add console stat sink
  ConsoleStatSink* console_sink = new ConsoleStatSink();
  System::stat_aggregator->RegisterSink(console_sink);
  stat_sinks_.push_back(console_sink);

  // Initialize stat aggregator
  System::stat_aggregator->Initialize();

  return SUCCESS;
}

CN24_SHELL_FUNC_IMPL(NetworkUnload) {
  CN24_SHELL_FUNC_DESCRIPTION("Unload the current network");
  CN24_SHELL_PARSE_ARGS;
  
  if(state_ == NOTHING) {
    LOGERROR << "There is no network loaded currently";
    return FAILURE;
  } else {
    // Unload trainer first
    if(state_ == NET_AND_TRAINER_LOADED) {
      delete trainer_;
      trainer_ = nullptr;
      state_ = NET_LOADED;

      // Generate any remaining stats
      System::stat_aggregator->StopRecording();
      System::stat_aggregator->Generate();

      // Delete stat aggregator and sinks
      delete System::stat_aggregator;
      for(unsigned int i = 0; i < stat_sinks_.size(); i++) {
        delete stat_sinks_[i];
      }
      stat_sinks_.clear();

      // Create new aggregator
      System::stat_aggregator = new StatAggregator();
    }
    
    // Create new vectors for bundle areas
    training_bundles_ = new std::vector<Bundle*>();
    training_weights_ = new std::vector<datum>();
    staging_bundles_ = new std::vector<Bundle*>();
    testing_bundles_ = new std::vector<Bundle*>();
    
    // Copy old data
    for(unsigned int i = 0; i < input_layer_->training_sets_.size(); i++)
      training_bundles_->push_back(input_layer_->training_sets_[i]);
    for(unsigned int i = 0; i < input_layer_->training_weights_.size(); i++)
      training_weights_->push_back(input_layer_->training_weights_[i]);
    for(unsigned int i = 0; i < input_layer_->staging_sets_.size(); i++)
      staging_bundles_->push_back(input_layer_->staging_sets_[i]);
    for(unsigned int i = 0; i < input_layer_->testing_sets_.size(); i++)
      testing_bundles_->push_back(input_layer_->testing_sets_[i]);
    
    // Destroy net
    delete graph_;
    graph_ = nullptr;
    input_layer_ = nullptr;
    
    delete class_manager_;
    class_manager_ = nullptr;
    
    state_ = NOTHING;
    return SUCCESS;
  }
}

CN24_SHELL_FUNC_IMPL(NetworkStatus) {
  CN24_SHELL_FUNC_DESCRIPTION("Displays information about the current network");

  int detailed_status = -1;

  cargo_add_option(cargo, (cargo_option_flags_t)0, "-v --detailed-status",
                   "Plots detailed information on every layer",
                   "b", &detailed_status);

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

  if(detailed_status && (state_ == NET_LOADED || state_ == NET_AND_TRAINER_LOADED)) {
    LOGINFO << "Doing forward pass for statistics...";
    graph_->FeedForward();
    for(NetGraphNode* node: graph_->GetNodes()) {
      LOGINFO << "Node \"" << node->unique_name << "\", " << node->layer->GetLayerDescription();
      for(NetGraphBuffer& buffer: node->output_buffers) {
        LOGINFO << "  Buffer \"" << buffer.description << "\"";
        LOGINFO << "    Data Size: " << buffer.combined_tensor->data;
        unsigned int nans = 0;
        for(unsigned int e = 0; e < buffer.combined_tensor->data.elements(); e++) {
          datum element = buffer.combined_tensor->data(e);
          if(!std::isfinite(element)) { nans++; }
        }
        LOGINFO << "    NaNs: " << nans << " of " << buffer.combined_tensor->data.elements();
      }
    }
  }
  return SUCCESS;
}

}