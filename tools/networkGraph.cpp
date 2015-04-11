/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
/**
 * @file networkGraph.cpp
 * @brief Writes a graphviz file displaying the network's architecture to standard output.
 *
 * @author Clemens-Alexander Brust(ikosa dot de at gmail dot com)
 */

#define NO_LOG_AT_ALL

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

int main (int argc, char* argv[]) {
  if (argc < 3) {
    LOGERROR << "USAGE: " << argv[0] << " <dataset config file> <net config file>";
    LOGEND;
    return -1;
  }

  std::string net_config_fname (argv[2]);
  std::string dataset_config_fname (argv[1]);

	std::ostringstream ss;

  Conv::System::Init();

  // Open network and dataset configuration files
  std::ifstream net_config_file (net_config_fname, std::ios::in);
  std::ifstream dataset_config_file (dataset_config_fname, std::ios::in);

  if (!net_config_file.good()) {
    FATAL ("Cannot open net configuration file!");
  }

  net_config_fname = net_config_fname.substr (net_config_fname.rfind ("/") + 1);

  if (!dataset_config_file.good()) {
    FATAL ("Cannot open dataset configuration file!");
  }

  dataset_config_fname = dataset_config_fname.substr (net_config_fname.rfind ("/") + 1);

  // Parse network configuration file
  Conv::ConfigurableFactory* factory = new Conv::ConfigurableFactory (net_config_file, 8347734, true);
  factory->InitOptimalSettings();
  
  // Extract important settings from parsed configuration
  Conv::TrainerSettings settings = factory->optimal_settings();
	settings.pbatchsize = 1;
  unsigned int BATCHSIZE = settings.pbatchsize;
  LOGDEBUG << "Optimal settings: " << settings;

  // Load dataset
  Conv::Dataset* dataset = nullptr;
	if (factory->method() == Conv::PATCH) {
		dataset = Conv::TensorStreamPatchDataset::CreateFromConfiguration(dataset_config_file, false, Conv::LOAD_BOTH, factory->patchsizex(), factory->patchsizey());
	}
	else if (factory->method() == Conv::FCN) {
		dataset = Conv::TensorStreamDataset::CreateFromConfiguration(dataset_config_file, false, Conv::LOAD_BOTH);
	}

  unsigned int CLASSES = dataset->GetClasses();

  // Assemble net
  Conv::Net net;
  int data_layer_id = 0;

  Conv::DatasetInputLayer* data_layer = nullptr;

	data_layer = new Conv::DatasetInputLayer (*dataset, BATCHSIZE, 1.0, 983923);
	data_layer_id = net.AddLayer (data_layer);

  int output_layer_id =
    factory->AddLayers (net, Conv::Connection (data_layer_id), CLASSES, true, ss);

  LOGDEBUG << "Output layer id: " << output_layer_id;

	Conv::NetGraphNode* data_node = new Conv::NetGraphNode(data_layer);

	Conv::NetGraph graph;
	graph.AddNode(data_node);
	bool completeness = factory->AddLayers(graph, Conv::NetGraphConnection(data_node, 0), CLASSES, true);

	graph.Initialize();

	LOGINFO << "Complete: " << completeness;

  LOGINFO << "DONE!";
  LOGEND;

	std::cout << "\ndigraph G {\n";
	graph.PrintGraph(std::cout);
	std::cout << "}\n";
	//std::cout << "\nGraph output:\ndigraph G {\n" << ss.str() << "\n}\n";
  return 0;
}