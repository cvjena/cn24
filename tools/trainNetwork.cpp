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

#include <private/NKContext.h>

void addStatLayers(Conv::NetGraph& graph, Conv::NetGraphNode* input_node, Conv::Dataset* dataset, Conv::ClassManager* class_manager);
bool parseCommand (Conv::ClassManager& class_manager, std::vector<Conv::Dataset*>& datasets, Conv::NetGraph& graph, Conv::NetGraph& testing_graph, Conv::Trainer& trainer, Conv::Trainer& testing_trainer, std::string& command);
void help();

int main (int argc, char* argv[]) {
  bool FROM_SCRIPT = false;
  bool NO_INIT = false;
  int requested_log_level = -1;
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
    FATAL("Gradient check is now part of the test suite!");
  } else if ((argc > 3 && std::string(argv[3]).compare("noinit") == 0) ||
    (argc > 4 && std::string(argv[4]).compare("noinit") == 0)) {
    NO_INIT = true;
    LOGINFO << "Skipping initialization...";
    argc--;
  }
  if (argc > 3) {
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
  
  // Open network and dataset configuration files
  std::ifstream* net_config_file = new std::ifstream(net_config_fname, std::ios::in);
  if (!net_config_file->good()) {
    FATAL ("Cannot open net configuration file!");
  }

  net_config_fname = net_config_fname.substr (net_config_fname.rfind ("/") + 1);
  
  // Parse network configuration file
  LOGDEBUG << "Parsing network config file..." << std::flush;
  Conv::JSONNetGraphFactory* factory = new Conv::JSONNetGraphFactory (*net_config_file, 8347734);

  // Open dataset configuration file
  std::ifstream dataset_config_file (dataset_config_fname, std::ios::in);

  if (!dataset_config_file.good()) {
    FATAL ("Cannot open dataset configuration file!");
  }

  dataset_config_fname = dataset_config_fname.substr (dataset_config_fname.rfind ("/") + 1);


  // Extract parallel batch size from parsed configuration
  unsigned int batch_size_parallel = 1;
  if(factory->GetHyperparameters().count("batch_size_parallel") == 1 && factory->GetHyperparameters()["batch_size_parallel"].is_number()) {
    batch_size_parallel = factory->GetHyperparameters()["batch_size_parallel"];
  }

  // Load dataset
  Conv::ClassManager class_manager;
  LOGINFO << "Loading dataset, this can take a long time depending on the size!" << std::flush;

  Conv::Dataset* initial_dataset = Conv::JSONDatasetFactory::ConstructDataset(Conv::JSON::parse(dataset_config_file), &class_manager);
  std::vector<Conv::Dataset*> datasets;
  datasets.push_back(initial_dataset);

  // Assemble net
  Conv::NetGraph graph;
  Conv::DatasetInputLayer* data_layer = nullptr;
	Conv::NetGraphNode* input_node = nullptr;

  data_layer = new Conv::DatasetInputLayer (factory->GetDataInput(), initial_dataset, batch_size_parallel, loss_sampling_p, 983923);
  input_node = new Conv::NetGraphNode(data_layer);
  input_node->is_input = true;
  graph.AddNode(input_node);

	bool completeness = factory->AddLayers(graph, &class_manager);
	LOGDEBUG << "Graph complete: " << completeness;
  
  if(!completeness)
    FATAL("Graph completeness test failed after factory run!");

	addStatLayers(graph, input_node, initial_dataset, &class_manager);
  
  if(!completeness)
    FATAL("Graph completeness test failed after adding stat layer!");

  // Initialize net with random weights
	graph.Initialize();

  graph.InitializeWeights(NO_INIT);

  Conv::Trainer trainer (graph, factory->GetHyperparameters());

  Conv::NetGraph* testing_graph;
  Conv::Trainer* testing_trainer;

  testing_graph = &graph;
  testing_trainer = &trainer;

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

      if (!parseCommand (class_manager, datasets, graph, *testing_graph, trainer, *testing_trainer, command) || script_file.eof())
        break;
    }
  } else {
    LOGINFO << "Enter \"help\" for information on how to use this program";

    while (true) {
      std::cout << "\n > " << std::flush;
      std::string command;
      std::getline (std::cin, command);

      if (!parseCommand (class_manager, datasets, graph, *testing_graph, trainer, *testing_trainer, command))
        break;
    }
  }

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}

void addStatLayers(Conv::NetGraph& graph, Conv::NetGraphNode* input_node, Conv::Dataset* dataset, Conv::ClassManager* class_manager) {
  if(dataset->GetTask() == Conv::SEMANTIC_SEGMENTATION || dataset->GetTask() == Conv::CLASSIFICATION) {
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
  } else if(dataset->GetTask() == Conv::DETECTION) {
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


bool parseCommand (Conv::ClassManager& class_manager, std::vector<Conv::Dataset*>& datasets_o, Conv::NetGraph& graph, Conv::NetGraph& testing_graph, Conv::Trainer& trainer, Conv::Trainer& testing_trainer, std::string& command) {
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

    unsigned int all = 0;
    unsigned int layerview = 0;
    Conv::ParseCountIfPossible(command, "view", layerview);
    Conv::ParseCountIfPossible(command, "all", all);
    testing_graph.SetLayerViewEnabled (layerview == 1);
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
          testing_trainer.SetEpoch(trainer.epoch());
          testing_trainer.Test();
          Conv::System::stat_aggregator->StopRecording();
          Conv::System::stat_aggregator->Generate();
          Conv::System::stat_aggregator->Reset();
        }
        // Restore old testing dataset
        input_layer->SetActiveTestingDataset(old_active_testing_dataset);
      }
    } else {
      Conv::System::stat_aggregator->StartRecording();
      testing_trainer.SetEpoch(trainer.epoch());
      testing_trainer.Test();
      Conv::System::stat_aggregator->StopRecording();
      Conv::System::stat_aggregator->Generate();
      Conv::System::stat_aggregator->Reset();
    }
    testing_graph.SetLayerViewEnabled(false);

  } else if (command.compare (0, 4, "load") == 0) {
    std::string param_file_name;
    Conv::ParseStringParamIfPossible (command, "file", param_file_name);

    if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    } else {
      std::ifstream param_file (param_file_name, std::ios::in | std::ios::binary);

      if (param_file.good()) {
        graph.DeserializeParameters (param_file);
        LOGINFO << "Loaded parameters from " << param_file_name;
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
  else if (command.compare(0, 5, "dump ") == 0) {
    std::string node_uid;
		Conv::ParseStringParamIfPossible(command, "node", node_uid);

    std::string param_file_name;
    Conv::ParseStringParamIfPossible (command, "file", param_file_name);

    if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    } else {
      std::ofstream param_file(param_file_name, std::ios::out | std::ios::binary);

      for (Conv::NetGraphNode *node : graph.GetNodes()) {
        if (node->unique_name.compare(node_uid) == 0) {
          for (Conv::CombinedTensor *param_tensor : node->layer->parameters()) {
            param_tensor->data.Serialize(param_file, true);
          }
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
  else if (command.compare(0, 7, "dsload ") == 0) {
    std::string dataset_filename;
    unsigned int restrict_samples = 0;
    Conv::ParseStringParamIfPossible(command, "file", dataset_filename);
    Conv::ParseCountIfPossible(command, "restrict", restrict_samples);

    // Open dataset configuration file
    std::ifstream dataset_config_file (dataset_filename, std::ios::in);

    if (!dataset_config_file.good()) {
      FATAL ("Cannot open dataset configuration file!");
    }

    Conv::Dataset* dataset = Conv::JSONDatasetFactory::ConstructDataset(Conv::JSON::parse(dataset_config_file), &class_manager);
    if(restrict_samples > 0) {
      LOGINFO << "Restricting training samples to first " << restrict_samples;
      dataset->RestrictTrainingSamples(restrict_samples);
    }
    
    Conv::DatasetInputLayer *input_layer = dynamic_cast<Conv::DatasetInputLayer *>(graph.GetInputNodes()[0]->layer);
    if(input_layer != nullptr) {
      LOGINFO << "Active testing dataset: " << input_layer->GetActiveTestingDataset()->GetName();
      input_layer->AddDataset(dataset, 1);

      LOGINFO << "Currently loaded datasets:";
      for(unsigned int d = 0; d < input_layer->GetDatasets().size(); d++) {
        LOGINFO << "  Dataset " << d << ": " << input_layer->GetDatasets()[d]->GetName();
      }
    }
  }
  else if (command.compare(0,6, "dslist") == 0) {
    LOGINFO << "Currently loaded datasets:";
    Conv::DatasetInputLayer *input_layer = dynamic_cast<Conv::DatasetInputLayer *>(graph.GetInputNodes()[0]->layer);
    if(input_layer != nullptr) {
      for (unsigned int d = 0; d < input_layer->GetDatasets().size(); d++) {
        LOGINFO << "  Dataset " << d << ": " << input_layer->GetDatasets()[d]->GetName();
        LOGINFO << "    Weight: " << input_layer->GetWeights()[d];
      }
      LOGINFO << "Active testing dataset: " << input_layer->GetActiveTestingDataset()->GetName();
    }
  }
  else if (command.compare(0,10,"dsselectt ") == 0) {
    unsigned int id = 0;
    Conv::ParseCountIfPossible(command, "id", id);
    Conv::DatasetInputLayer *input_layer = dynamic_cast<Conv::DatasetInputLayer *>(graph.GetInputNodes()[0]->layer);
    if(input_layer != nullptr) {
      if (id < input_layer->GetDatasets().size()) {
        Conv::DatasetInputLayer *input_layer = dynamic_cast<Conv::DatasetInputLayer *>(graph.GetInputNodes()[0]->layer);
        if (input_layer != nullptr) {
          input_layer->SetActiveTestingDataset(input_layer->GetDatasets()[id]);
        } else {
          LOGERROR << "Cannot find dataset input layer";
        }
      } else {
        LOGERROR << "Dataset " << id << " does not exist!";
      }
    }
  }
  else if (command.compare(0,12,"dssetweight ") == 0) {
    unsigned int id = 0;
    Conv::datum weight = 0;
    Conv::ParseCountIfPossible(command, "id", id);
    Conv::ParseDatumParamIfPossible(command, "weight", weight);
    Conv::DatasetInputLayer *input_layer = dynamic_cast<Conv::DatasetInputLayer *>(graph.GetInputNodes()[0]->layer);
    if(input_layer != nullptr) {
      if (id < input_layer->GetDatasets().size()) {
        Conv::DatasetInputLayer *input_layer = dynamic_cast<Conv::DatasetInputLayer *>(graph.GetInputNodes()[0]->layer);
        if (input_layer != nullptr) {
          input_layer->SetWeight(input_layer->GetDatasets()[id], weight);
        } else {
          LOGERROR << "Cannot find dataset input layer";
        }
      } else {
        LOGERROR << "Dataset " << id << " does not exist!";
      }
    }
  }
  else if (command.compare(0,7,"explore") == 0) {
    Conv::NetGraphNode* input_node = graph.GetInputNodes()[0];

    Conv::DatasetInputLayer *input_layer = dynamic_cast<Conv::DatasetInputLayer *>(input_node->layer);
    if(input_layer != nullptr) {
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

            if(input_layer->GetActiveTestingDataset()->GetTask() == Conv::DETECTION) {
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
            if(input_layer->GetActiveTestingDataset()->GetTask() == Conv::DETECTION) {
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
