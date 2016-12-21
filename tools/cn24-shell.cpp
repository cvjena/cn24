/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
/**
 * @file cn24-shell.cpp
 * @brief cn24 command line
 *
 * @author Clemens-Alexander Brust(ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <ctime>
#include <cstring>

#include <cn24.h>
#include <private/ConfigParsing.h>

#include <private/NKContext.h>

void addStatLayers(Conv::NetGraph& graph, Conv::NetGraphNode* input_node, Conv::Task task, Conv::ClassManager* class_manager);
bool parseCommand (Conv::ClassManager& class_manager, Conv::SegmentSetInputLayer* input_layer, Conv::NetGraph& graph, Conv::Trainer& trainer, std::string& command);
void exploreData(const Conv::ClassManager &class_manager, Conv::SegmentSetInputLayer *input_layer, Conv::NetGraph &graph);
void showWeightStats(Conv::NetGraph &graph, const std::string &command);
void help();

void train(Conv::NetGraph &graph, Conv::Trainer &trainer, const std::string &command);

void test(Conv::NetGraph &graph, Conv::Trainer &trainer, const std::string &command);

void loadModel(Conv::NetGraph &graph, const std::string &command);

void saveModel(Conv::NetGraph &graph, const std::string &command);

void setExperimentProperty(const std::string &command);

void setEpoch(Conv::Trainer &trainer, const std::string &command);

void resetTrainer(Conv::NetGraph &graph, Conv::Trainer &trainer);

void dumpNetGraph(Conv::NetGraph &graph, const std::string &command);

void dumpLayerData(Conv::NetGraph &graph, const std::string &command);

void showDataBufferStats(Conv::NetGraph &graph, const std::string &command);

void setTrainerStats(Conv::Trainer &trainer, const std::string &command);

void displaySegmentSetsInfo(const std::vector<Conv::SegmentSet *> &sets);

Conv::SegmentSet *findSegmentSet(const Conv::SegmentSetInputLayer *input_layer, const std::string &set_name);

int main (int argc, char* argv[]) {
  bool FROM_SCRIPT = false;
  int requested_log_level = -1;
  const Conv::datum loss_sampling_p = 0.5;
  
  if(argc > 1) {
    if(std::string(argv[1]).compare("-v") == 0) {
      requested_log_level = 3;
      argv[1] = argv[0];
      argc--; argv++;
    }
  }
  

  if (argc < 2) {
    LOGERROR << "USAGE: " << argv[0] << " [-v] <net config file> [script file]";
    LOGEND;
    return -1;
  }

  std::string script_fname;

  if (argc > 2) {
    FROM_SCRIPT = true;
    script_fname = argv[2];
  }

  std::string net_config_fname (argv[1]);

  Conv::System::Init(requested_log_level);
  
  // Register stat sinks
  Conv::ConsoleStatSink console_stat_sink;
  Conv::CSVStatSink csv_stat_sink;
  Conv::System::stat_aggregator->RegisterSink(&console_stat_sink);
  Conv::System::stat_aggregator->RegisterSink(&csv_stat_sink);
  
  // Open network and dataset configuration files
  std::ifstream* net_config_file = new std::ifstream(Conv::PathFinder::FindPath(net_config_fname, {}), std::ios::in);
  if (!net_config_file->good()) {
    FATAL ("Cannot open net configuration file!");
  }

  net_config_fname = net_config_fname.substr (net_config_fname.rfind ("/") + 1);
  
  // Parse network configuration file
  LOGDEBUG << "Parsing network config file..." << std::flush;
  Conv::JSONNetGraphFactory* factory = new Conv::JSONNetGraphFactory (*net_config_file, 8347734);

  // Extract parallel batch size from parsed configuration
  unsigned int batch_size_parallel = 1;
  if(factory->GetHyperparameters().count("batch_size_parallel") == 1 && factory->GetHyperparameters()["batch_size_parallel"].is_number()) {
    batch_size_parallel = factory->GetHyperparameters()["batch_size_parallel"];
  }

  Conv::ClassManager class_manager;

  // Assemble net
  Conv::SegmentSetInputLayer* input_layer = nullptr;
  Conv::NetGraph graph;
	Conv::NetGraphNode* input_node = nullptr;

  input_layer = new Conv::SegmentSetInputLayer (factory->GetDataInput(), Conv::DETECTION, &class_manager, batch_size_parallel, 983923);
  input_node = new Conv::NetGraphNode(input_layer);
  input_node->is_input = true;
  graph.AddNode(input_node);

	bool completeness = factory->AddLayers(graph, &class_manager);
	LOGDEBUG << "Graph complete: " << completeness;
  
  if(!completeness)
    FATAL("Graph completeness test failed after factory run!");

	addStatLayers(graph, input_node, Conv::DETECTION, &class_manager);
  
  if(!completeness)
    FATAL("Graph completeness test failed after adding stat layer!");

  // Assemble initial segment sets
  Conv::SegmentSet* default_training_set = new Conv::SegmentSet("Default_Training");
  input_layer->training_sets_.push_back(default_training_set);
  input_layer->training_weights_.push_back(1);
  Conv::SegmentSet* default_testing_set = new Conv::SegmentSet("Default_Testing");
  input_layer->testing_sets_.push_back(default_testing_set);
  input_layer->UpdateDatasets();

  // Initialize net with random weights
	graph.Initialize();
  graph.InitializeWeights();

  Conv::Trainer trainer (graph, factory->GetHyperparameters());

  Conv::System::stat_aggregator->Initialize();
  Conv::System::stat_aggregator->SetCurrentTestingDataset(0);
  LOGINFO << "Current training settings: " << factory->GetHyperparameters().dump();

  if (FROM_SCRIPT) {
    LOGINFO << "Executing script: " << script_fname;
    std::ifstream script_file (script_fname, std::ios::in);

    if (!script_file.good()) {
      FATAL ("Cannot open " << script_fname);
    }

    while (true) {
      std::string command;
      std::getline (script_file, command);

      if(command.compare(0, 5, "shell") == 0) {
        goto shell_part;
      }
      if (!parseCommand (class_manager, input_layer, graph, trainer, command) || script_file.eof()) {
        break;
      }
    }
  } else {
    shell_part:
    LOGINFO << "Enter \"help\" for information on how to use this program";

    while (true) {
      std::cout << "\n > " << std::flush;
      std::string command;
      std::getline (std::cin, command);

      if (!parseCommand (class_manager, input_layer, graph, trainer, command))
        break;
    }
  }

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}

void addStatLayers(Conv::NetGraph& graph, Conv::NetGraphNode* input_node, Conv::Task task, Conv::ClassManager* class_manager) {
  if(task == Conv::SEMANTIC_SEGMENTATION || task == Conv::CLASSIFICATION) {
    for (Conv::NetGraphNode *output_node : graph.GetOutputNodes()) {
      // Add appropriate statistics layer
      Conv::NetGraphNode *stat_node = nullptr;
      if (class_manager->GetClassCount() == 1) {
        Conv::BinaryStatLayer *binary_stat_layer = new Conv::BinaryStatLayer(13, -1, 1);
        stat_node = new Conv::NetGraphNode(binary_stat_layer);
      } else {
        Conv::ConfusionMatrixLayer *confusion_matrix_layer = new Conv::ConfusionMatrixLayer(class_manager);
        stat_node = new Conv::NetGraphNode(confusion_matrix_layer);
      }
      stat_node->input_connections.push_back(Conv::NetGraphConnection(output_node, 0, false));
      stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 1));
      stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 3));
      graph.AddNode(stat_node);
    }
  } else if(task == Conv::DETECTION) {
    for (Conv::NetGraphNode *output_node : graph.GetOutputNodes()) {
      // Add appropriate statistics layer
      Conv::NetGraphNode *stat_node = nullptr;
      Conv::DetectionStatLayer *detection_stat_layer = new Conv::DetectionStatLayer(class_manager);

      stat_node = new Conv::NetGraphNode(detection_stat_layer);
      stat_node->input_connections.push_back(Conv::NetGraphConnection(output_node, 0, false));
      stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 1));
      stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node, 3));
      graph.AddNode(stat_node);
    }
  }
}


bool parseCommand (Conv::ClassManager& class_manager, Conv::SegmentSetInputLayer* input_layer, Conv::NetGraph& graph, Conv::Trainer& trainer, std::string& command) {
  if (command.compare ("q") == 0 || command.compare ("quit") == 0) {
    return false;
  } else if (command.compare (0, 5, "train") == 0) {
    train(graph, trainer, command);
  } else if (command.compare (0, 4, "test") == 0) {
    test(graph, trainer, command);
  } else if (command.compare (0, 4, "load") == 0) {
    loadModel(graph, command);
  } else if (command.compare (0, 4, "save") == 0) {
    saveModel(graph, command);
  } else if (command.compare (0, 4, "seg-") == 0) {
    // Segment operation
    std::string seg_command = command.substr(4);
    if(seg_command.compare(0, 5, "move ") == 0) {
      std::string source_set_name, segment_name, target_set_name;
      Conv::ParseStringParamIfPossible(seg_command, "source", source_set_name);
      Conv::ParseStringParamIfPossible(seg_command, "target", target_set_name);
      Conv::ParseStringParamIfPossible(seg_command, "segment", segment_name);

      Conv::SegmentSet *source_set = findSegmentSet(input_layer, source_set_name);
      Conv::SegmentSet *target_set = findSegmentSet(input_layer, target_set_name);

      if(source_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      } else if(target_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << target_set_name << "\"";
      } else {
        int segment_index = source_set->GetSegmentIndex(segment_name);
        if(segment_index >= 0) {
          Conv::Segment* segment = source_set->GetSegment((unsigned int)segment_index);
          source_set->RemoveSegment((unsigned int)segment_index);
          target_set->AddSegment(segment);
          input_layer->UpdateDatasets();
          LOGINFO << "Moved segment \"" << segment->name << "\" from SegmentSet \"" << source_set->name << "\" to \"" << target_set->name << "\"";
        } else {
          LOGWARN << "Could not find segment \"" << segment_name << "\" in SegmentSet \"" << source_set->name << "\"";
        }
      }
    } else if(seg_command.compare(0, 8, "movebest") == 0) {
      std::string source_set_name, target_set_name;
      Conv::ParseStringParamIfPossible(seg_command, "source", source_set_name);
      Conv::ParseStringParamIfPossible(seg_command, "target", target_set_name);

      Conv::SegmentSet *source_set = findSegmentSet(input_layer, source_set_name);
      Conv::SegmentSet *target_set = findSegmentSet(input_layer, target_set_name);

      if(source_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      } else if(target_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << target_set_name << "\"";
      } else {
        if(source_set->GetSegmentCount() > 0) {
          Conv::datum max_score = source_set->GetSegment(0)->score;
          unsigned int max_index = 0;
          for (unsigned int s = 0; s < source_set->GetSegmentCount(); s++) {
            Conv::datum score = source_set->GetSegment(s)->score;
            if(score > max_score) {
              max_score = score; max_index = s;
            }
          }
          Conv::Segment *segment = source_set->GetSegment(max_index);
          target_set->AddSegment(segment);
          source_set->RemoveSegment(max_index);

          input_layer->UpdateDatasets();
          LOGINFO << "Moved segment \"" << segment->name << "\" from SegmentSet \"" << source_set->name << "\" to \"" << target_set->name << "\"";
        } else {
          LOGWARN << "There are no segments in SegmentSet \"" << source_set->name << "\"";
        }
      }
    } else if(seg_command.compare(0, 7, "moveall") == 0) {
      std::string source_set_name, target_set_name;
      Conv::ParseStringParamIfPossible(seg_command, "source", source_set_name);
      Conv::ParseStringParamIfPossible(seg_command, "target", target_set_name);

      Conv::SegmentSet *source_set = findSegmentSet(input_layer, source_set_name);
      Conv::SegmentSet *target_set = findSegmentSet(input_layer, target_set_name);

      if(source_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      } else if(target_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << target_set_name << "\"";
      } else {
        while(source_set->GetSegmentCount() > 0) {
          Conv::Segment* segment = source_set->GetSegment(0);
          target_set->AddSegment(segment);
          source_set->RemoveSegment(0);
        }
        input_layer->UpdateDatasets();
        LOGINFO << "Moved all segments from SegmentSet \"" << source_set->name << "\" to \"" << target_set->name << "\"";
      }
    } else if(seg_command.compare(0, 5, "split") == 0) {
      std::string source_set_name, segment_name, target_set_name;
      unsigned int bucket_size;
      Conv::ParseStringParamIfPossible(seg_command, "source", source_set_name);
      Conv::ParseStringParamIfPossible(seg_command, "target", target_set_name);
      Conv::ParseStringParamIfPossible(seg_command, "segment", segment_name);
      Conv::ParseCountIfPossible(seg_command, "size", bucket_size);

      Conv::SegmentSet *source_set = findSegmentSet(input_layer, source_set_name);
      Conv::SegmentSet *target_set = findSegmentSet(input_layer, target_set_name);

      if(source_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      } else if(target_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << target_set_name << "\"";
      } else {
        if(bucket_size > 0) {
          int segment_index = source_set->GetSegmentIndex(segment_name);
          if(segment_index >= 0) {
            Conv::Segment* segment = source_set->GetSegment((unsigned int)segment_index);
            source_set->RemoveSegment((unsigned int)segment_index);

            unsigned int split_segment_index = 0;
            for(unsigned int start_sample = 0; start_sample < segment->GetSampleCount(); start_sample+=bucket_size) {
              std::stringstream ss; ss << segment->name << "_" << split_segment_index;
              Conv::Segment* split_segment = new Conv::Segment(ss.str());
              for(unsigned int sample = 0; sample < bucket_size && (start_sample + sample) < segment->GetSampleCount(); sample++) {
                split_segment->AddSample(segment->GetSample(start_sample + sample), {}, true);
              }
              target_set->AddSegment(split_segment);
              split_segment_index++;
            }

            LOGINFO << "Split segment \"" << segment->name << "\"";
            input_layer->UpdateDatasets();
          } else {
            LOGWARN << "Could not find segment \"" << segment_name << "\" in SegmentSet \"" << source_set->name << "\"";
          }
        } else {
          LOGWARN << "Bucket size needs to be at least 1";
        }
      }
    } else {
      LOGWARN << "Unknown segment command: " << seg_command;
    }
  } else if (command.compare (0, 4, "set-") == 0) {
    // SegmentSet operation
    std::string set_command = command.substr(4);
    if(set_command.compare(0, 4, "load") == 0) {
      std::string filename, folder_hint;
      Conv::ParseStringParamIfPossible(set_command, "file", filename);
      Conv::ParseStringParamIfPossible(set_command, "hint", folder_hint);
      std::string resolved_path = Conv::PathFinder::FindPath(filename, {});
      std::ifstream set_file(resolved_path, std::ios::in);
      if(set_file.good()) {
        Conv::SegmentSet *set = new Conv::SegmentSet("Unnamed SegmentSet");
        bool success = set->Deserialize(Conv::JSON::parse(set_file), folder_hint);
        if(!success) {
          LOGERROR << "Deserialization failed!";
          LOGERROR << "Could not open " << filename << " (" << resolved_path << ")";
          delete set;
        } else {
          LOGINFO << "Loaded SegmentSet \"" << set->name << "\" (" << set->GetSampleCount() << " samples).";
          input_layer->staging_sets_.push_back(set);
          input_layer->UpdateDatasets();
        }
      } else {
        LOGERROR << "Could not open " << filename << " (" << resolved_path << ")";
      }
    } else if(set_command.compare(0, 5, "score") == 0) {
      std::string source_set_name;
      Conv::ParseStringParamIfPossible(set_command, "name", source_set_name);
      Conv::SegmentSet *source_set = findSegmentSet(input_layer, source_set_name);

      if(source_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      } else {
        for(unsigned int s = 0; s < source_set->GetSegmentCount(); s++) {
          Conv::Segment* segment = source_set->GetSegment(s);
          segment->score = 1;
        }
        LOGINFO << "Finished scoring SegmentSet \"" << source_set->name << "\"";
      }
    } else if(set_command.compare(0, 3, "new") == 0) {
      std::string name = "Unnamed SegmentSet";
      Conv::ParseStringParamIfPossible(set_command, "name", name);
      Conv::SegmentSet* set = new Conv::SegmentSet(name);

      input_layer->staging_sets_.push_back(set);
      input_layer->UpdateDatasets();
    } else if(set_command.compare(0, 4, "list") == 0) {
      LOGINFO << "SegmentSets (TRAINING):";
      displaySegmentSetsInfo(input_layer->training_sets_);
      LOGINFO << "Weights (TRAINING):";
      for(unsigned int set = 0; set < input_layer->training_sets_.size(); set++) {
        LOGINFO << "  \"" << input_layer->training_sets_[set]->name << "\": " << input_layer->training_weights_[set];
      }
      LOGINFO << "SegmentSets (STAGING):";
      displaySegmentSetsInfo(input_layer->staging_sets_);
      LOGINFO << "SegmentSets (TESTING):";
      displaySegmentSetsInfo(input_layer->testing_sets_);
    } else {
      LOGWARN << "Unknown set command: " << set_command;
    }
  } else if (command.compare (0, 14, "set experiment") == 0) {
    setExperimentProperty(command);
  } else if (command.compare (0, 9, "set epoch") == 0) {
    setEpoch(trainer, command);
  } else if (command.compare (0, 5, "reset") == 0) {
    resetTrainer(graph, trainer);
  } else if (command.compare (0, 4, "help") == 0) {
    help();
	} else if (command.compare (0, 5, "graph") == 0) {
    dumpNetGraph(graph, command);
  } else if (command.compare(0, 5, "wstat") == 0) {
    showWeightStats(graph, command);
  } else if (command.compare(0, 5, "dump ") == 0) {
    dumpLayerData(graph, command);
  } else if (command.compare(0, 5, "dstat") == 0) {
    showDataBufferStats(graph, command);
  } else if (command.compare(0, 5, "tstat") == 0) {
    setTrainerStats(trainer, command);
  } else if (command.compare(0,7,"explore") == 0) {
    exploreData(class_manager, input_layer, graph);
  } else {
    LOGWARN << "Unknown command: " << command;
  }

  return true;
}

Conv::SegmentSet *findSegmentSet(const Conv::SegmentSetInputLayer *input_layer, const std::string &set_name) {
  Conv::SegmentSet* set = nullptr;
  for(Conv::SegmentSet* set_ : input_layer->training_sets_) {
        if(set_->name.compare(set_name) == 0) { set = set_; }
      }
  for(Conv::SegmentSet* set_ : input_layer->staging_sets_) {
        if(set_->name.compare(set_name) == 0) { set = set_; }
      }
  for(Conv::SegmentSet* set_ : input_layer->testing_sets_) {
        if(set_->name.compare(set_name) == 0) { set = set_; }
      }
  return set;
}

void displaySegmentSetsInfo(const std::vector<Conv::SegmentSet *> &sets) {
  for(Conv::SegmentSet* set : sets) {
        LOGINFO << "  \"" << set->name << "\" (" << set->GetSegmentCount() << " segments, " << set->GetSampleCount() << " samples):";
        for(unsigned int seg = 0; seg < set->GetSegmentCount(); seg++) {
          Conv::Segment* segment = set->GetSegment(seg);
          LOGINFO << "    \"" << segment->name << "\" (" << segment->GetSampleCount() << " samples, score: " << segment->score << ")";
        }
      }
}

void setTrainerStats(Conv::Trainer &trainer, const std::string &command) {
  unsigned int enable_tstat = 1;
  Conv::ParseCountIfPossible(command, "enable", enable_tstat);
  trainer.SetStatsDuringTraining(enable_tstat == 1);
  LOGDEBUG << "Training stats enabled: " << enable_tstat;
}

void showDataBufferStats(Conv::NetGraph &graph, const std::string &command) {
  std::string node_uid;
  Conv::ParseStringParamIfPossible(command, "node", node_uid);
  for (Conv::NetGraphNode* node : graph.GetNodes()) {
			if (node->unique_name.compare(node_uid) == 0) {
				for (Conv::NetGraphBuffer& output_buffer : node->output_buffers) {
					Conv::CombinedTensor* output_tensor = output_buffer.combined_tensor;
					LOGINFO << "Reporting stats on buffer " << output_buffer.description;
					LOGINFO << "Data stats:";
					output_tensor->data.PrintStats();
					LOGINFO << "Delta stats:";
					output_tensor->delta.PrintStats();
				}
			}
		}
}

void dumpLayerData(Conv::NetGraph &graph, const std::string &command) {
  std::string node_uid;
  Conv::ParseStringParamIfPossible(command, "node", node_uid);

  std::string param_file_name;
  Conv::ParseStringParamIfPossible (command, "file", param_file_name);

  if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    } else {
      std::ofstream param_file(param_file_name, std::ios_base::out | std::ios_base::binary);

      for (Conv::NetGraphNode *node : graph.GetNodes()) {
        if (node->unique_name.compare(node_uid) == 0) {
          for (Conv::CombinedTensor *param_tensor : node->layer->parameters()) {
            param_tensor->data.Serialize(param_file, true);
          }
        }
      }
    }
}

void dumpNetGraph(Conv::NetGraph &graph, const std::string &command) {
  std::string param_file_name;
  Conv::ParseStringParamIfPossible (command, "file", param_file_name);

  if (param_file_name.length() == 0) {
			LOGERROR << "Filename needed!";
		}
		else {
			std::ofstream graph_output(param_file_name, std::ios_base::out);
			graph_output << "digraph G {";
      graph.PrintGraph(graph_output);
			graph_output << "}";
			graph_output.close();
		}
}

void resetTrainer(Conv::NetGraph &graph, Conv::Trainer &trainer) {
  LOGINFO << "Resetting parameters";
  graph.InitializeWeights();
  trainer.Reset();
}

void setEpoch(Conv::Trainer &trainer, const std::string &command) {
  unsigned int epoch = 0;
  Conv::ParseCountIfPossible (command, "epoch", epoch);
  LOGINFO << "Setting current epoch to " << epoch;
  trainer.SetEpoch (epoch);
  trainer.Reset();
}

void setExperimentProperty(const std::string &command) {
  std::string experiment_name = "";
  Conv::ParseStringParamIfPossible(command, "name", experiment_name);
  if(experiment_name.length() > 0) {
      Conv::System::stat_aggregator->SetCurrentExperiment(experiment_name);
  } else {
      LOGINFO << "Experiment name not specified, not changing!";
    }
}

void saveModel(Conv::NetGraph &graph, const std::string &command) {
  std::string param_file_name;
  Conv::ParseStringParamIfPossible (command, "file", param_file_name);

  if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    } else {
      std::ofstream param_file (param_file_name, std::ios_base::out | std::ios_base::binary);

      if (param_file.good()) {
        graph.SerializeParameters (param_file);
        LOGINFO << "Written parameters to " << param_file_name;
      } else {
        LOGERROR << "Cannot open " << param_file_name;
      }

      param_file.close();
    }
}

void loadModel(Conv::NetGraph &graph, const std::string &command) {
  std::string param_file_name;
  Conv::ParseStringParamIfPossible (command, "file", param_file_name);

  if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    } else {
      std::ifstream param_file (param_file_name, std::ios_base::in | std::ios_base::binary);

      if (param_file.good()) {
        graph.DeserializeParameters (param_file);
        LOGINFO << "Loaded parameters from " << param_file_name;
      } else {
        LOGERROR << "Cannot open " << param_file_name;
      }

      param_file.close();
    }
}

void test(Conv::NetGraph &graph, Conv::Trainer &trainer, const std::string &command) {
  unsigned int all = 0;
  unsigned int layerview = 0;
  Conv::ParseCountIfPossible(command, "view", layerview);
  Conv::ParseCountIfPossible(command, "all", all);
  graph.SetLayerViewEnabled (layerview == 1);
  if(all == 1) {
      // Test all datasets
      Conv::DatasetInputLayer *input_layer = dynamic_cast<Conv::DatasetInputLayer *>(graph.GetInputNodes()[0]->layer);
      if(input_layer != nullptr) {
        // Save old testing dataset
        Conv::Dataset* old_active_testing_dataset = input_layer->GetActiveTestingDataset();
        for (unsigned int d = 0; d < input_layer->GetDatasets().size(); d++) {
          // Test each dataset
          input_layer->SetActiveTestingDataset(input_layer->GetDatasets()[d]);
          Conv::System::stat_aggregator->StartRecording();
          trainer.Test();
          Conv::System::stat_aggregator->StopRecording();
          Conv::System::stat_aggregator->Generate();
          Conv::System::stat_aggregator->Reset();
        }
        // Restore old testing dataset
        input_layer->SetActiveTestingDataset(old_active_testing_dataset);
      }
    } else {
      Conv::System::stat_aggregator->StartRecording();
      trainer.Test();
      Conv::System::stat_aggregator->StopRecording();
      Conv::System::stat_aggregator->Generate();
      Conv::System::stat_aggregator->Reset();
    }
  graph.SetLayerViewEnabled(false);
}

void train(Conv::NetGraph &graph, Conv::Trainer &trainer, const std::string &command) {
  Conv::System::stat_aggregator->StartRecording();

  unsigned int epochs = 1;
  unsigned int layerview = 0;
  unsigned int no_snapshots = 0;
  Conv::ParseCountIfPossible (command, "view", layerview);
  graph.SetLayerViewEnabled (layerview == 1);
  Conv::ParseCountIfPossible (command, "epochs", epochs);
  Conv::ParseCountIfPossible(command, "no_snapshots", no_snapshots);
  trainer.Train (epochs, no_snapshots != 1);
  graph.SetLayerViewEnabled (false);
  LOGINFO << "Training complete.";

  Conv::System::stat_aggregator->StopRecording();
  if(no_snapshots == 1)
      Conv::System::stat_aggregator->Generate();
  Conv::System::stat_aggregator->Reset();
}

void showWeightStats(Conv::NetGraph &graph, const std::string &command) {
  std::string node_uid;
  unsigned int show = 0;
  unsigned int map = 0;
  unsigned int sample = 0;
  Conv::ParseStringParamIfPossible(command, "node", node_uid);
  Conv::ParseCountIfPossible(command, "show", show);
  Conv::ParseCountIfPossible(command, "map", map);
  Conv::ParseCountIfPossible(command, "sample", sample);

  for (Conv::NetGraphNode* node : graph.GetNodes()) {
    if (node->unique_name.compare(node_uid) == 0) {
      unsigned int p = 0;
      for (Conv::CombinedTensor* param_tensor : node->layer->parameters()) {
        if(show == 1) {
          Conv::System::viewer->show(&(param_tensor->data), "Tensor Viewer", false, map, sample);
        } else {
          LOGINFO << "Reporting stats on parameter set " << p++ << " " << param_tensor->data;
          LOGINFO << "Weight stats:";
          param_tensor->data.PrintStats();
          LOGINFO << "Gradient stats:";
          param_tensor->delta.PrintStats();
        }
      }
    }
  }
}

void exploreData(const Conv::ClassManager &class_manager, Conv::SegmentSetInputLayer *input_layer, Conv::NetGraph &graph) {
  Conv::NetGraphNode* input_node = graph.GetInputNodes()[0];
  input_layer->SelectAndLoadSamples();
  graph.OnBeforeFeedForward();
  std::vector<Conv::NetGraphNode*> input_nodes = {input_node};
  graph.FeedForward(input_nodes, true);
  Conv::NetGraphBuffer& output_buffer = input_node->output_buffers[0];
  Conv::NetGraphBuffer& label_buffer = input_node->output_buffers[1];
  // Conv::System::viewer->show(...)
  {
      Conv::NKContext context{};
      int current_sample = 0;
      Conv::NKImage data_image(context, output_buffer.combined_tensor->data, current_sample);
      Conv::NKImage label_image(context, label_buffer.combined_tensor->data, current_sample);
      while(true) {
        context.ProcessEvents();
        if (nk_begin(context, "Data Tensor", nk_rect(0, 0, 500, 600),
                     NK_WINDOW_TITLE | NK_WINDOW_CLOSABLE | NK_WINDOW_SCALABLE | NK_WINDOW_MOVABLE)) {
          const unsigned int output_width = output_buffer.combined_tensor->data.width();
          const unsigned int output_height = output_buffer.combined_tensor->data.height();

          nk_layout_row_dynamic(context, 30, 1);
          nk_property_int(context, "Sample", 0, &current_sample, output_buffer.combined_tensor->data.samples() - 1, 1, 0.01);
          data_image.SetSample(current_sample);
          nk_layout_row_static(context, output_buffer.combined_tensor->data.height(), output_buffer.combined_tensor->data.width(), 1);
          nk_image(context, data_image);

          if(input_layer->GetTask() == Conv::DETECTION) {
            std::vector<Conv::BoundingBox>* boxes = (std::vector<Conv::BoundingBox>*)label_buffer.combined_tensor->metadata[current_sample];
            for(unsigned int b = 0; b < boxes->size(); b++) {
              Conv::BoundingBox bbox = boxes->at(b);
              struct nk_rect bbox_rect = nk_layout_space_rect_to_screen(context, nk_rect(4.0 + (bbox.x - (bbox.w/2.0f)) * (float)output_width, (bbox.y - (bbox.h/2.0f)) * (float)output_height, bbox.w * (float)output_width, bbox.h*(float)output_height));
              struct nk_rect text_rect = bbox_rect;
              text_rect.y = bbox_rect.y + bbox_rect.h - 12.0f;
              text_rect.h = 12.0f;
              nk_stroke_rect(nk_window_get_canvas(context), bbox_rect,1, 1, nk_rgb(255,255,255));
              nk_draw_text(nk_window_get_canvas(context), text_rect,
                class_manager.GetClassInfoById(bbox.c).first.c_str(),
                class_manager.GetClassInfoById(bbox.c).first.length(), context.context_->style.font, nk_rgb(255,255,255), nk_rgb(0,0,0));
            }
          }

          nk_layout_row_static(context, label_buffer.combined_tensor->data.height(), label_buffer.combined_tensor->data.width(), 1);
          nk_image(context, label_image);

        }
        nk_end(context);
        if(nk_begin(context, "Metadata", nk_rect(501, 0, 200, 600),
                    NK_WINDOW_TITLE | NK_WINDOW_MOVABLE)) {
          nk_layout_row_dynamic(context, 30, 1);
          if(nk_button_label(context, "Select new samples")){
            input_layer->SelectAndLoadSamples();
            data_image.Update();
            label_image.Update();
          }
          if(input_layer->GetTask() == Conv::DETECTION) {
            nk_layout_row_dynamic(context, 30, 2);
            std::vector<Conv::BoundingBox>* boxes = (std::vector<Conv::BoundingBox>*)label_buffer.combined_tensor->metadata[current_sample];
            for(unsigned int b = 0; b < boxes->size(); b++) {
              Conv::BoundingBox bbox = boxes->at(b);
              nk_label(context, class_manager.GetClassInfoById(bbox.c).first.c_str(), NK_TEXT_ALIGN_LEFT);
            }
          }
        }
        nk_end(context);
        if(nk_window_is_closed(context, "Data Tensor"))
          break;
        context.Draw();
      }
    }
}

void help() {
  std::cout << "You can use the following commands:\n";
  std::cout
      << "  train [epochs=<n>] [no_snapshots=1]\n"
      << "    Train the network for n epochs (default: 1). no_snapshots=1 accumulates statistics over all n epochs.\n\n"
      << "  test\n"
      << "    Test the network\n\n"
      << "  set epoch=<epoch>\n"
      << "    Sets the current epoch\n\n"
      << "  set experiment name=<name>\n"
      << "    Sets the current experiment name for logging and statistics purposes\n\n"
      << "  reset\n"
      << "    Reinitializes the nets parameters\n\n"
      << "  load file=<path> [last_layer=<l>]\n"
      << "    Load parameters from a file for all layers up to l (default: all layers)\n\n"
			<< "  graph file=<path> {test|train}\n"
			<< "    Write the network architecture for training/testing to a file in graphviz format\n\n"
      << "  save file=<path>\n"
      << "    Save parameters to a file\n\n"
      << "  tstat enable=<1|0>\n"
      << "    Enable statistics during training (1: yes, 0: no)\n";
}
