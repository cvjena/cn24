/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
/**
 * @file trainNetwork.cpp
 * @brief Trains a convolutional neural net for prediction.
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

void addStatLayers(Conv::NetGraph& graph, Conv::NetGraphNode* input_node, Conv::Dataset* dataset);
bool parseCommand (Conv::NetGraph& graph, Conv::NetGraph& testing_graph, Conv::Trainer& trainer, Conv::Trainer& testing_trainer, bool hybrid, std::string& command);
void help();

int main (int argc, char* argv[]) {
  bool GRADIENT_CHECK = false;
  bool FROM_SCRIPT = false;
  int requested_log_level = -1;
#ifdef LAYERTIME
  const Conv::datum it_factor = 0.01;
#else
  const Conv::datum it_factor = 1;
#endif
  const Conv::datum loss_sampling_p = 0.5;
  
  if(argc > 1) {
    if(std::string(argv[1]).compare("-v") == 0) {
      requested_log_level = 3;
      argv[1] = argv[0];
      argc--; argv++;
    }
  }
  

  if (argc < 3) {
    LOGERROR << "USAGE: " << argv[0] << " [-v] <dataset config file> <net config file> {[script file]|gradient_check}";
    LOGEND;
    return -1;
  }

  std::string script_fname;

  if (argc > 3 && std::string (argv[3]).compare ("gradient_check") == 0) {
    GRADIENT_CHECK = true;
  } else if (argc > 3) {
    FROM_SCRIPT = true;
    script_fname = argv[3];
  }

  std::string net_config_fname (argv[2]);
  std::string dataset_config_fname (argv[1]);

  Conv::System::Init(requested_log_level);
  
  // Register stat sinks
  Conv::ConsoleStatSink console_stat_sink;
  Conv::CSVStatSink csv_stat_sink;
  Conv::System::stat_aggregator->RegisterSink(&console_stat_sink);
  Conv::System::stat_aggregator->RegisterSink(&csv_stat_sink);
  
  Conv::Factory* factory;
  
  // Open network and dataset configuration files
  std::ifstream* net_config_file = new std::ifstream(net_config_fname, std::ios::in);
  if (!net_config_file->good()) {
    FATAL ("Cannot open net configuration file!");
  }

  net_config_fname = net_config_fname.substr (net_config_fname.rfind ("/") + 1);
  
  // Parse network configuration file
  factory = new Conv::ConfigurableFactory (*net_config_file, 8347734, true);

  // Open dataset configuration file
  std::ifstream dataset_config_file (dataset_config_fname, std::ios::in);

  if (!dataset_config_file.good()) {
    FATAL ("Cannot open dataset configuration file!");
  }

  dataset_config_fname = dataset_config_fname.substr (net_config_fname.rfind ("/") + 1);

  factory->InitOptimalSettings();
  
  // Extract important settings from parsed configuration
  const bool patchwise_training = (factory->method() == Conv::PATCH);
  unsigned int BATCHSIZE = factory->optimal_settings().pbatchsize;
  
  LOGINFO << "Using " << (patchwise_training ? "hybrid patchwise" : "fully convolutional") << " training";

  Conv::TrainerSettings settings = factory->optimal_settings();
  settings.epoch_training_ratio = 1 * it_factor;
  settings.testing_ratio = 1 * it_factor;

  // Load dataset
  LOGINFO << "Loading dataset, this can take a long time depending on the size!" << std::flush;
  Conv::Dataset* dataset = nullptr;

  if (patchwise_training) {
    dataset = Conv::TensorStreamPatchDataset::CreateFromConfiguration (dataset_config_file, false, (patchwise_training && !GRADIENT_CHECK) ? Conv::LOAD_TRAINING_ONLY : Conv::LOAD_BOTH,
              factory->patchsizex(), factory->patchsizey());
  } else {
    dataset = Conv::TensorStreamDataset::CreateFromConfiguration (dataset_config_file, false, Conv::LOAD_BOTH);
  }

  unsigned int CLASSES = dataset->GetClasses();

  // Assemble net
  Conv::NetGraph graph;
  Conv::DatasetInputLayer* data_layer = nullptr;
	Conv::NetGraphNode* input_node = nullptr;

  if (GRADIENT_CHECK) {
		Conv::Tensor* data_tensor;
    Conv::Tensor* weight_tensor;
    Conv::Tensor* label_tensor;
    Conv::Tensor* helper_tensor;
		if (patchwise_training) {
			data_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), dataset->GetInputMaps());
			weight_tensor = new Conv::Tensor(BATCHSIZE, 1, 1, 1);
			label_tensor = new Conv::Tensor(BATCHSIZE, 1, 1, dataset->GetLabelMaps());
			helper_tensor = new Conv::Tensor(BATCHSIZE, 1, 1, 2);
		} else {
			data_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), dataset->GetInputMaps());
			weight_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), 1);
			label_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), dataset->GetLabelMaps());
			helper_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), 2);
		}

		bool dataset_success = true;
    for (unsigned int b = 0; b < BATCHSIZE; b++)
      dataset_success &= dataset->GetTestingSample (*data_tensor, *label_tensor, *helper_tensor, *weight_tensor, b, b);

		if (!dataset_success) {
			FATAL("Could not load samples for gradient check!");
		}
    Conv::InputLayer* input_layer = new Conv::InputLayer (*data_tensor, *label_tensor, *helper_tensor, *weight_tensor);
		input_node = new Conv::NetGraphNode(input_layer);
		input_node->is_input = true;
		graph.AddNode(input_node);
  } else {
    data_layer = new Conv::DatasetInputLayer (*dataset, BATCHSIZE, patchwise_training ? 1.0 : loss_sampling_p, 983923);
		input_node = new Conv::NetGraphNode(data_layer);
		input_node->is_input = true;
		graph.AddNode(input_node);
  }

	bool completeness = factory->AddLayers(graph, Conv::NetGraphConnection(input_node), CLASSES, true);
	LOGDEBUG << "Graph complete: " << completeness;
  
  if(!completeness)
    FATAL("Graph completeness test failed after factory run!");

	addStatLayers(graph, input_node, dataset);
  
  if(!completeness)
    FATAL("Graph completeness test failed after adding stat layer!");

  // Initialize net with random weights
	graph.Initialize();
  graph.InitializeWeights();

  if (GRADIENT_CHECK) {
    Conv::GradientTester::TestGradient (graph);
  } else {
    Conv::Trainer trainer (graph, settings);

    Conv::NetGraph* testing_graph;
    Conv::Trainer* testing_trainer;

    if (patchwise_training) {
      // This overrides the batch size for testing in this scope
      unsigned int BATCHSIZE = 1;
      
      // Assemble testing net
      Conv::TensorStreamDataset* testing_dataset = Conv::TensorStreamDataset::CreateFromConfiguration (dataset_config_file, false, Conv::LOAD_TESTING_ONLY);
      testing_graph = new Conv::NetGraph();

      Conv::DatasetInputLayer* tdata_layer = nullptr;
      tdata_layer = new Conv::DatasetInputLayer (*testing_dataset, BATCHSIZE, 1.0, 983923);
      Conv::NetGraphNode* tinput_node = new Conv::NetGraphNode(tdata_layer);
      tinput_node->is_input = true;
      testing_graph->AddNode(tinput_node);

      Conv::ConfigurableFactory* tfactory = new Conv::ConfigurableFactory (*net_config_file, 8347734);
      bool testing_completeness = tfactory->AddLayers(*testing_graph, Conv::NetGraphConnection(tinput_node), CLASSES, true);
      LOGDEBUG << "Testing graph complete: " << testing_completeness;

      if(!completeness)
        FATAL("Graph completeness test failed after factory run!");

      addStatLayers(*testing_graph, tinput_node, testing_dataset);
      
      if(!completeness)
        FATAL("Graph completeness test failed after adding stat layer!");

      testing_graph->Initialize();

      // Shadow training net weights
      std::vector<Conv::CombinedTensor*> training_params;
      std::vector<Conv::CombinedTensor*> testing_params;
      graph.GetParameters (training_params);
      testing_graph->GetParameters (testing_params);

      for (unsigned int p = 0; p < training_params.size(); p++) {
        Conv::CombinedTensor* training_ct = training_params[p];
        Conv::CombinedTensor* testing_ct = testing_params[p];
        testing_ct->data.Shadow (training_ct->data);
        testing_ct->delta.Shadow (training_ct->delta);
      }

      Conv::TrainerSettings settings = tfactory->optimal_settings();
      settings.pbatchsize = 1;
      settings.sbatchsize = 1;
      testing_trainer = new Conv::Trainer (*testing_graph, settings);
    } else {
      testing_graph = &graph;
      testing_trainer = &trainer;
    }

    Conv::System::stat_aggregator->Initialize();
    LOGINFO << "Current training settings: " << factory->optimal_settings();

    if (FROM_SCRIPT) {
      LOGINFO << "Executing script: " << script_fname;
      std::ifstream script_file (script_fname, std::ios::in);

      if (!script_file.good()) {
        FATAL ("Cannot open " << script_fname);
      }

      while (true) {
        std::string command;
        std::getline (script_file, command);

        if (!parseCommand (graph, *testing_graph, trainer, *testing_trainer, patchwise_training, command) || script_file.eof())
          break;
      }
    } else {
      LOGINFO << "Enter \"help\" for information on how to use this program";

      while (true) {
        std::cout << "\n > " << std::flush;
        std::string command;
        std::getline (std::cin, command);

        if (!parseCommand (graph, *testing_graph, trainer, *testing_trainer, patchwise_training, command))
          break;
      }
    }
  }

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}

void addStatLayers(Conv::NetGraph& graph, Conv::NetGraphNode* input_node, Conv::Dataset* dataset) {
	for (Conv::NetGraphNode* output_node : graph.GetOutputNodes()) {
		// Add appropriate statistics layer
		Conv::NetGraphNode* stat_node = nullptr;
		if (dataset->GetClasses() == 1) {
			Conv::BinaryStatLayer* binary_stat_layer = new Conv::BinaryStatLayer (13, -1, 1);
			stat_node = new Conv::NetGraphNode(binary_stat_layer);
		} else {
			std::vector<std::string> class_names = dataset->GetClassNames();
			Conv::ConfusionMatrixLayer* confusion_matrix_layer = new Conv::ConfusionMatrixLayer (class_names, dataset->GetClasses());
			stat_node = new Conv::NetGraphNode(confusion_matrix_layer);
		}
		stat_node->input_connections.push_back(Conv::NetGraphConnection(output_node, 0, false));
		stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node,1));
		stat_node->input_connections.push_back(Conv::NetGraphConnection(input_node,3));
		graph.AddNode(stat_node);
	}
}


bool parseCommand (Conv::NetGraph& graph, Conv::NetGraph& testing_graph, Conv::Trainer& trainer, Conv::Trainer& testing_trainer, bool hybrid, std::string& command) {
  if (command.compare ("q") == 0 || command.compare ("quit") == 0) {
    return false;
  } else if (command.compare (0, 5, "train") == 0) {
    Conv::System::stat_aggregator->StartRecording();
    
    unsigned int epochs = 1;
    unsigned int layerview = 0;
    unsigned int no_snapshots = 0;
    Conv::ParseCountIfPossible (command, "view", layerview);
    graph.SetLayerViewEnabled (layerview == 1);
    Conv::ParseCountIfPossible (command, "epochs", epochs);
    Conv::ParseCountIfPossible(command, "no_snapshots", no_snapshots);
    trainer.Train (epochs, no_snapshots != 1);
    testing_trainer.SetEpoch (trainer.epoch());
    graph.SetLayerViewEnabled (false);
    LOGINFO << "Training complete.";
    
    Conv::System::stat_aggregator->StopRecording();
    if(no_snapshots == 1)
      Conv::System::stat_aggregator->Generate();
    Conv::System::stat_aggregator->Reset();
  } else if (command.compare (0, 4, "test") == 0) {
    Conv::System::stat_aggregator->StartRecording();
    
    unsigned int layerview = 0;
    Conv::ParseCountIfPossible (command, "view", layerview);
    testing_graph.SetLayerViewEnabled (layerview == 1);
    testing_trainer.SetEpoch (trainer.epoch());
    testing_trainer.Test();
    testing_graph.SetLayerViewEnabled (false);
    LOGINFO << "Testing complete.";
    
    Conv::System::stat_aggregator->StopRecording();
    Conv::System::stat_aggregator->Generate();
    Conv::System::stat_aggregator->Reset();
  } else if (command.compare (0, 4, "load") == 0) {
    std::string param_file_name;
    unsigned int last_layer = 0;
    Conv::ParseStringParamIfPossible (command, "file", param_file_name);
    Conv::ParseCountIfPossible (command, "last_layer", last_layer);

    if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    } else {
      std::ifstream param_file (param_file_name, std::ios::in | std::ios::binary);

      if (param_file.good()) {
        graph.DeserializeParameters (param_file, last_layer);
        LOGINFO << "Loaded parameters from " << param_file_name;

        if (hybrid) {
          LOGDEBUG << "Reshadowing tensors...";
          // Shadow training net weights
          std::vector<Conv::CombinedTensor*> training_params;
          std::vector<Conv::CombinedTensor*> testing_params;
          graph.GetParameters (training_params);
          testing_graph.GetParameters (testing_params);

          for (unsigned int p = 0; p < training_params.size(); p++) {
            Conv::CombinedTensor* training_ct = training_params[p];
            Conv::CombinedTensor* testing_ct = testing_params[p];
            testing_ct->data.Shadow (training_ct->data);
            testing_ct->delta.Shadow (training_ct->delta);
          }
        }
      } else {
        LOGERROR << "Cannot open " << param_file_name;
      }

      param_file.close();
    }
  } else if (command.compare (0, 4, "save") == 0) {
    std::string param_file_name;
    Conv::ParseStringParamIfPossible (command, "file", param_file_name);

    if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    } else {
      std::ofstream param_file (param_file_name, std::ios::out | std::ios::binary);

      if (param_file.good()) {
        graph.SerializeParameters (param_file);
        LOGINFO << "Written parameters to " << param_file_name;
      } else {
        LOGERROR << "Cannot open " << param_file_name;
      }

      param_file.close();
    }
  } else if (command.compare (0, 14, "set experiment") == 0) {
    std::string experiment_name = "";
    Conv::ParseStringParamIfPossible(command, "name", experiment_name);
    if(experiment_name.length() > 0)
      Conv::System::stat_aggregator->SetCurrentExperiment(experiment_name);
    else
      LOGINFO << "Experiment name not specified, not changing!";
  } else if (command.compare (0, 9, "set epoch") == 0) {
    unsigned int epoch = 0;
    Conv::ParseCountIfPossible (command, "epoch", epoch);
    LOGINFO << "Setting current epoch to " << epoch;
    trainer.SetEpoch (epoch);
    testing_trainer.SetEpoch (trainer.epoch());
    trainer.Reset();
  } else if (command.compare (0, 5, "reset") == 0) {
    LOGINFO << "Resetting parameters";
    graph.InitializeWeights();
    trainer.Reset();

    if (hybrid) {
      LOGDEBUG << "Reshadowing tensors...";
      // Shadow training net weights
      std::vector<Conv::CombinedTensor*> training_params;
      std::vector<Conv::CombinedTensor*> testing_params;
      graph.GetParameters (training_params);
      testing_graph.GetParameters (testing_params);

      for (unsigned int p = 0; p < training_params.size(); p++) {
        Conv::CombinedTensor* training_ct = training_params[p];
        Conv::CombinedTensor* testing_ct = testing_params[p];
        testing_ct->data.Shadow (training_ct->data);
        testing_ct->delta.Shadow (training_ct->delta);
      }
    }
  } else if (command.compare (0, 4, "help") == 0) {
    help();
	} else if (command.compare (0, 5, "graph") == 0) {
    std::string param_file_name;
    Conv::ParseStringParamIfPossible (command, "file", param_file_name);

		if (param_file_name.length() == 0) {
			LOGERROR << "Filename needed!";
		}
		else {
			std::ofstream graph_output(param_file_name, std::ios::out);
			graph_output << "digraph G {";
			if (command.find("test") != std::string::npos) {
				testing_graph.PrintGraph(graph_output);
			} else {
				graph.PrintGraph(graph_output);
			}
			graph_output << "}";
			graph_output.close();
		}
	}
	else if (command.compare(0, 5, "wstat") == 0) {
    std::string node_uid;
		Conv::ParseStringParamIfPossible(command, "node", node_uid);
		for (Conv::NetGraphNode* node : graph.GetNodes()) {
			if (node->unique_name.compare(node_uid) == 0) {
				unsigned int p = 0;
				for (Conv::CombinedTensor* param_tensor : node->layer->parameters()) {
					LOGINFO << "Reporting stats on parameter set " << p++ << " " << param_tensor->data;
					LOGINFO << "Weight stats:";
					param_tensor->data.PrintStats();
					LOGINFO << "Gradient stats:";
					param_tensor->delta.PrintStats();
				}
			}
		}
	}
	else if (command.compare(0, 5, "dstat") == 0) {
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
	else if (command.compare(0, 5, "tstat") == 0) {
    unsigned int enable_tstat = 1;
    Conv::ParseCountIfPossible(command, "enable", enable_tstat);
    trainer.SetStatsDuringTraining(enable_tstat == 1);
    testing_trainer.SetStatsDuringTraining(enable_tstat == 1);
    LOGDEBUG << "Training stats enabled: " << enable_tstat;
  }
	else {
    LOGWARN << "Unknown command: " << command;
  }

  return true;
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
