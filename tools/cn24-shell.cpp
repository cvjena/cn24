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

void test(Conv::SegmentSetInputLayer *input_layer, Conv::NetGraph &graph, Conv::Trainer &trainer, const std::string &command);

void loadModel(Conv::NetGraph &graph, const std::string &command);

void saveModel(Conv::NetGraph &graph, const std::string &command);

void setExperimentProperty(const std::string &command);

void setEpoch(Conv::Trainer &trainer, const std::string &command);

void resetTrainer(Conv::NetGraph &graph, Conv::Trainer &trainer);

void dumpNetGraph(Conv::NetGraph &graph, const std::string &command);

void dumpLayerData(Conv::NetGraph &graph, const std::string &command);

void showDataBufferStats(Conv::NetGraph &graph, const std::string &command);

void setTrainerStats(Conv::Trainer &trainer, const std::string &command);

void displaySegmentSetsInfo(const std::vector<Conv::Bundle *> &sets);

Conv::Bundle *findSegmentSet(const Conv::SegmentSetInputLayer *input_layer, const std::string &set_name);

int stat_id_correct_pred;
int stat_id_correct_loc;
int stat_id_wrong_pred;

int main (int argc, char* argv[]) {
  bool FROM_SCRIPT = false;
  int requested_log_level = -1;

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

  // Register stats
  Conv::StatDescriptor desc_correct_loc;
  desc_correct_loc.description = "Correctly Localized Predictions";
  desc_correct_loc.nullable = true;
  desc_correct_loc.init_function =
      [](Conv::Stat& stat) {stat.is_null = true; stat.value = 0.0;};
  desc_correct_loc.unit = "%";
  desc_correct_loc.update_function =
    [](Conv::Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
  desc_correct_loc.output_function =
    [](Conv::HardcodedStats& hc_stats, Conv::Stat& stat) -> Conv::Stat {
      Conv::Stat return_stat; return_stat.is_null = true;
      if (hc_stats.iterations > 0 && !stat.is_null) {
        double d_iterations = (double)hc_stats.iterations;
        return_stat.value = 100.0 * stat.value / d_iterations;
        return_stat.is_null = false;
      }
      return return_stat;
    };

  Conv::StatDescriptor desc_correct_pred;
  desc_correct_pred.description = "Completely Correct Predictions";
  desc_correct_pred.nullable = true;
  desc_correct_pred.init_function =
      [](Conv::Stat& stat) {stat.is_null = true; stat.value = 0.0;};
  desc_correct_pred.unit = "%";
  desc_correct_pred.update_function =
    [](Conv::Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
  desc_correct_pred.output_function =
    [](Conv::HardcodedStats& hc_stats, Conv::Stat& stat) -> Conv::Stat {
      Conv::Stat return_stat; return_stat.is_null = true;
      if (hc_stats.iterations > 0 && !stat.is_null) {
        double d_iterations = (double)hc_stats.iterations;
        return_stat.value = 100.0 * stat.value / d_iterations;
        return_stat.is_null = false;
      }
      return return_stat;
    };

  Conv::StatDescriptor desc_wrong_pred;
  desc_wrong_pred.description = "Wrong Predictions";
  desc_wrong_pred.nullable = true;
  desc_wrong_pred.init_function =
      [](Conv::Stat& stat) {stat.is_null = true; stat.value = 0.0;};
  desc_wrong_pred.unit = "%";
  desc_wrong_pred.update_function =
    [](Conv::Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
  desc_wrong_pred.output_function =
    [](Conv::HardcodedStats& hc_stats, Conv::Stat& stat) -> Conv::Stat {
      Conv::Stat return_stat; return_stat.is_null = true;
      if (hc_stats.iterations > 0 && !stat.is_null) {
        double d_iterations = (double)hc_stats.iterations;
        return_stat.value = 100.0 * stat.value / d_iterations;
        return_stat.is_null = false;
      }
      return return_stat;
    };

  stat_id_correct_pred = Conv::System::stat_aggregator->RegisterStat(&desc_correct_pred);
  stat_id_correct_loc = Conv::System::stat_aggregator->RegisterStat(&desc_correct_loc);
  stat_id_wrong_pred = Conv::System::stat_aggregator->RegisterStat(&desc_wrong_pred);

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
  Conv::Bundle* default_training_set = new Conv::Bundle("Default_Training");
  input_layer->training_sets_.push_back(default_training_set);
  input_layer->training_weights_.push_back(1);
  Conv::Bundle* default_testing_set = new Conv::Bundle("Default_Testing");
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
  } else if (command.compare (0, 7, "meminfo") == 0) {
    LOGINFO << "There are " << Conv::System::registry->size() << " tensors "
    << "registered.";
    std::cout << std::endl;
    std::cout << std::setw(20) << "Dimensions";
    std::cout << std::setw(12) << "Bytes used";
    std::cout << std::setw(5) << "GPU";
    std::cout << std::setw(15) << "Construction";
    std::cout << std::setw(25) << "Owner";
    std::cout << std::setw(30) << "Comment" << std::endl;
    
    for(Conv::TensorRegistry::const_iterator it = Conv::System::registry->begin();
        it != Conv::System::registry->end(); it++) {
      Conv::Tensor* tensor = *it;
      std::cout << std::setw(20) << *tensor ;
      std::cout << std::setw(12) << tensor->elements() * sizeof(Conv::datum);
      std::cout << std::setw(5) << (tensor->cl_data_ptr_ == nullptr ? 
      "No" : (tensor->cl_gpu_ ? "Yes" : "Yes*"));
      std::cout << std::setw(15) << tensor->construction;
      std::cout << std::setw(25) << tensor->owner;
      std::cout << std::setw(30) << tensor->comment;
      std::cout << std::endl;
    }
  } else if (command.compare (0, 5, "train") == 0) {
    train(graph, trainer, command);
  } else if (command.compare (0, 4, "test") == 0) {
    test(input_layer, graph, trainer, command);
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

      Conv::Bundle *source_set = findSegmentSet(input_layer, source_set_name);
      Conv::Bundle *target_set = findSegmentSet(input_layer, target_set_name);

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

      Conv::Bundle *source_set = findSegmentSet(input_layer, source_set_name);
      Conv::Bundle *target_set = findSegmentSet(input_layer, target_set_name);

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

      Conv::Bundle *source_set = findSegmentSet(input_layer, source_set_name);
      Conv::Bundle *target_set = findSegmentSet(input_layer, target_set_name);

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

      Conv::Bundle *source_set = findSegmentSet(input_layer, source_set_name);
      Conv::Bundle *target_set = findSegmentSet(input_layer, target_set_name);

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
        Conv::Bundle *set = new Conv::Bundle("Unnamed SegmentSet");
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
    } else if(set_command.compare(0, 4, "test") == 0) {
      std::string source_set_name;
      Conv::ParseStringParamIfPossible(set_command, "name", source_set_name);
      bool found = false;
      for(unsigned int set = 0; set < input_layer->testing_sets_.size(); set++) {
        if(input_layer->testing_sets_[set]->name.compare(source_set_name) == 0) {
          input_layer->SetActiveTestingSet(set);
          LOGINFO << "Set SegmentSet \"" << input_layer->training_sets_[set]->name << "\" to be the active testing dataset";
          found = true;
          break;
        }
      }
      if(found) {
        input_layer->UpdateDatasets();
      } else {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      }

    } else if(set_command.compare(0, 6, "weight") == 0) {
      std::string source_set_name;
      Conv::datum weight = 1;
      Conv::ParseStringParamIfPossible(set_command, "name", source_set_name);
      Conv::ParseDatumParamIfPossible(set_command, "weight", weight);
      bool found = false;
      for(unsigned int set = 0; set < input_layer->training_sets_.size(); set++) {
        if(input_layer->training_sets_[set]->name.compare(source_set_name) == 0) {
          input_layer->training_weights_[set] = weight;
          LOGINFO << "Set weight of SegmentSet \"" << input_layer->training_sets_[set]->name << "\" to " << weight;
          found = true;
          break;
        }
      }
      if(found) {
        input_layer->UpdateDatasets();
      } else {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      }
    } else if(set_command.compare(0, 5, "score") == 0) {
      std::string source_set_name;
      Conv::ParseStringParamIfPossible(set_command, "name", source_set_name);
      Conv::Bundle *source_set = findSegmentSet(input_layer, source_set_name);

      if(source_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      } else {
        for(unsigned int s = 0; s < source_set->GetSegmentCount(); s++) {
          Conv::Segment* segment = source_set->GetSegment(s);
          segment->score = 1;
        }
        LOGINFO << "Finished scoring SegmentSet \"" << source_set->name << "\"";
      }
    } else if(set_command.compare(0, 4, "hypo") == 0) {
      std::string source_set_name;
      Conv::datum confidence_threshold = -1;
      Conv::ParseStringParamIfPossible(set_command, "name", source_set_name);
      Conv::ParseDatumParamIfPossible(set_command, "threshold", confidence_threshold);
      Conv::Bundle* source_set = findSegmentSet(input_layer, source_set_name);

      Conv::YOLODetectionLayer* detection_layer = dynamic_cast<Conv::YOLODetectionLayer*>(graph.GetOutputNodes()[0]->layer);
      if (detection_layer == nullptr) {
        LOGERROR << "Output node is not a YOLO detection layer!";
        return true;
      }

      const Conv::datum old_threshold = detection_layer->GetConfidenceThreshold();
      if (confidence_threshold > 0) {
        LOGDEBUG << "Setting confidence threshold to " << confidence_threshold;
        detection_layer->SetConfidenceThreshold(confidence_threshold);
      }

      Conv::NetGraphNode* input_node = graph.GetInputNodes()[0];
      Conv::NetGraphBuffer& label_buffer = input_node->output_buffers[1];
      Conv::NetGraphBuffer& prediction_buffer = graph.GetOutputNodes()[0]->output_buffers[0];
      Conv::DetectionMetadataPointer* label_metadata = (Conv::DetectionMetadataPointer*)label_buffer.combined_tensor->metadata;
      Conv::DetectionMetadataPointer* predicted_metadata = (Conv::DetectionMetadataPointer*)prediction_buffer.combined_tensor->metadata;

      if(source_set == nullptr) {
        LOGWARN << "Could not find SegmentSet \"" << source_set_name << "\"";
      } else {
        // Statistics
        Conv::System::stat_aggregator->StartRecording();
        unsigned int total_labels = 0;
        unsigned int correct_labels = 0;
        unsigned int total_predictions = 0;
        unsigned int correct_predictions = 0;
        unsigned int only_localised = 0;
        unsigned int wrong_predictions = 0;

        input_layer->ForceWeightsZero();
        graph.SetIsTesting(true);
        unsigned int batch_size = prediction_buffer.combined_tensor->data.samples();


        for(unsigned int s = 0; s < source_set->GetSegmentCount(); s++) {
          Conv::Segment* segment = source_set->GetSegment(s);
          LOGDEBUG << "Hypothesizing Segment \"" << segment->name << "\"";
          std::cout << std::endl << std::flush;

          for(unsigned int sample = 0; sample < segment->GetSampleCount(); sample+= batch_size) {
            for(unsigned int bindex = 0; bindex < batch_size && (sample+bindex) < segment->GetSampleCount(); bindex++) {
              // for(unsigned int sample = 0; sample < 1; sample++) {
              Conv::JSON& sample_json = segment->GetSample(sample + bindex);
              input_layer->ForceLoadDetection(sample_json, bindex);
            }
            graph.FeedForward();
            for(unsigned int bindex = 0; bindex < batch_size && (sample+bindex) < segment->GetSampleCount(); bindex++) {
              Conv::JSON& sample_json = segment->GetSample(sample + bindex);
              sample_json["original_boxes"] = sample_json["boxes"];
              sample_json["boxes"] = Conv::JSON::array();

              // Go through all labeled boxes to set flag1 to false
              for (unsigned int lbox = 0; lbox < label_metadata[bindex]->size(); lbox++) {
                Conv::BoundingBox &lbbox = label_metadata[bindex]->at(lbox);
                lbbox.flag1 = false;
                total_labels++;
              }

              // Go trough all predicted boxes
              for (unsigned int box = 0; box < predicted_metadata[bindex]->size(); box++) {
                Conv::BoundingBox &bbox = predicted_metadata[bindex]->at(box);
                bool found_box = false;

                // Compare to all labeled boxes
                for (unsigned int lbox = 0; lbox < label_metadata[bindex]->size(); lbox++) {
                  Conv::BoundingBox &lbbox = label_metadata[bindex]->at(lbox);
                  Conv::datum iou = lbbox.IntersectionOverUnion(&bbox);
                  if (iou > 0.5) {
                    // IoU is enough, this is a match
                    found_box = true;
                    lbbox.flag1 = true;

                    // Check classification
                    if (bbox.c == lbbox.c) {
                      correct_predictions++;
                    } else {
                      only_localised++;
                    }

                    // Add bounding box to sample
                    Conv::JSON bbox_json = Conv::JSON::object(); //sample_json["original_boxes"][lbox];
                    bbox_json["w"] = bbox.w;
                    bbox_json["h"] = bbox.h;
                    bbox_json["x"] = bbox.x;
                    bbox_json["y"] = bbox.y;
                    bbox_json["class"] = class_manager.GetClassInfoById(lbbox.c).first;
                    bbox_json["dont_scale"] = 1;
                    sample_json["boxes"].push_back(bbox_json);
                    break;
                  }
                }

                if (!found_box) {
                  wrong_predictions++;
                }
                total_predictions++;
              }

              // Go through all labeled boxes to reset flag
              for (unsigned int lbox = 0; lbox < label_metadata[bindex]->size(); lbox++) {
                Conv::BoundingBox &lbbox = label_metadata[bindex]->at(lbox);
                if(lbbox.flag1)
                  correct_labels++;
                lbbox.flag1 = false;
              }

              std::cout << "." << std::flush;
            }
          }

        }
        // Print statistics
        LOGDEBUG << "Correct predictions  : " << correct_predictions;
        LOGDEBUG << "Correct localisations: " << only_localised;
        LOGDEBUG << "Wrong predictions    : " << wrong_predictions;
        LOGDEBUG << "Total predicted boxes: " << total_predictions;
        LOGDEBUG << "---------------------------";
        LOGDEBUG << "Found labeled boxes  : " << correct_labels;
        LOGDEBUG << "Total labeled boxes  : " << total_labels;

        LOGINFO << "Finished hypothesizing SegmentSet \"" << source_set->name << "\"";

        Conv::System::stat_aggregator->hardcoded_stats_.iterations += total_predictions;

        Conv::System::stat_aggregator->Update(stat_id_correct_loc, only_localised);
        Conv::System::stat_aggregator->Update(stat_id_correct_pred, correct_predictions);
        Conv::System::stat_aggregator->Update(stat_id_wrong_pred, wrong_predictions);

        Conv::System::stat_aggregator->StopRecording();
        Conv::System::stat_aggregator->Generate();
        Conv::System::stat_aggregator->Reset();
      }

      detection_layer->SetConfidenceThreshold(old_threshold);
    } else if(set_command.compare(0, 4, "move") == 0) {
      std::string source_area, target_area, set_name;
      Conv::ParseStringParamIfPossible(set_command, "name", set_name);
      Conv::ParseStringParamIfPossible(set_command, "source", source_area);
      Conv::ParseStringParamIfPossible(set_command, "target", target_area);

      if(target_area.compare("training") == 0 || target_area.compare("staging") == 0 || target_area.compare("testing") == 0) {
        Conv::Bundle* segment_set = nullptr;
        if(source_area.compare("training") == 0) {
          for(unsigned int set = 0; set < input_layer->training_sets_.size(); set++) { if (input_layer->training_sets_[set]->name.compare(set_name) == 0) {
              segment_set = input_layer->training_sets_[set]; input_layer->training_sets_.erase(input_layer->training_sets_.begin() + set);
              input_layer->training_weights_.erase(input_layer->training_weights_.begin() + set);
              break;
            } }
        } else if(source_area.compare("staging") == 0) {
          for(unsigned int set = 0; set < input_layer->staging_sets_.size(); set++) { if (input_layer->staging_sets_[set]->name.compare(set_name) == 0) {
              segment_set = input_layer->staging_sets_[set]; input_layer->staging_sets_.erase(input_layer->staging_sets_.begin() + set);
              break;
            } }
        } else if(source_area.compare("testing") == 0) {
          for (unsigned int set = 0; set < input_layer->testing_sets_.size(); set++) { if (input_layer->testing_sets_[set]->name.compare(set_name) == 0) {
              segment_set = input_layer->testing_sets_[set]; input_layer->testing_sets_.erase(input_layer->testing_sets_.begin() + set);
              break;
            }
          }
        } else {
          LOGWARN << "Unknown source area \"" << source_area << "\"";
        }

        if(segment_set != nullptr) {
          if(target_area.compare("training") == 0) {
            input_layer->training_sets_.push_back(segment_set);
            input_layer->training_weights_.push_back(1);
          } else if(target_area.compare("staging") == 0) {
            input_layer->staging_sets_.push_back(segment_set);
          } else if(target_area.compare("testing") == 0) {
            input_layer->testing_sets_.push_back(segment_set);
          }
          LOGINFO << "Moved SegmentSet " << segment_set->name << " from " << source_area << " to " << target_area;
          input_layer->UpdateDatasets();
        } else {
          LOGWARN << "Unknown SegmentSet \"" << set_name << "\"";
        }
      } else {
        LOGWARN << "Unknown target area \"" << target_area << "\"";
      }
    } else if(set_command.compare(0, 3, "new") == 0) {
      std::string name = "Unnamed SegmentSet";
      Conv::ParseStringParamIfPossible(set_command, "name", name);
      Conv::Bundle* set = new Conv::Bundle(name);

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

Conv::Bundle *findSegmentSet(const Conv::SegmentSetInputLayer *input_layer, const std::string &set_name) {
  Conv::Bundle* set = nullptr;
  for(Conv::Bundle* set_ : input_layer->training_sets_) {
        if(set_->name.compare(set_name) == 0) { set = set_; }
      }
  for(Conv::Bundle* set_ : input_layer->staging_sets_) {
        if(set_->name.compare(set_name) == 0) { set = set_; }
      }
  for(Conv::Bundle* set_ : input_layer->testing_sets_) {
        if(set_->name.compare(set_name) == 0) { set = set_; }
      }
  return set;
}

void displaySegmentSetsInfo(const std::vector<Conv::Bundle *> &sets) {
  for(Conv::Bundle* set : sets) {
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

void test(Conv::SegmentSetInputLayer *input_layer, Conv::NetGraph &graph, Conv::Trainer &trainer, const std::string &command) {
  unsigned int all = 0;
  unsigned int layerview = 0;
  Conv::ParseCountIfPossible(command, "view", layerview);
  Conv::ParseCountIfPossible(command, "all", all);
  graph.SetLayerViewEnabled (layerview == 1);
  if(all == 1) {
    // Save old testing dataset
    unsigned int old_active_testing_set = input_layer->GetActiveTestingSet();
    for (unsigned int d = 0; d < input_layer->testing_sets_.size(); d++) {
      // Test each dataset
      input_layer->SetActiveTestingSet(d);
      Conv::System::stat_aggregator->StartRecording();
      trainer.Test();
      Conv::System::stat_aggregator->StopRecording();
      Conv::System::stat_aggregator->Generate();
      Conv::System::stat_aggregator->Reset();
    }
    // Restore old testing dataset
    input_layer->SetActiveTestingSet(old_active_testing_set);
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
