/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */


#include <string>
#include <vector>
#include <random>
#include <cn24/net/YOLOLossLayer.h>
#include "../net/DummyErrorLayer.h"
#include "../net/ErrorLayer.h"
#include "../net/YOLOLossLayer.h"
#include "../net/YOLODynamicOutputLayer.h"

#include "NetGraph.h"
#include "NetGraphNode.h"
#include "LayerFactory.h"

#include "JSONNetGraphFactory.h"

namespace Conv {

bool JSONNetGraphFactory::AddLayers(NetGraph &graph, ClassManager* class_manager, unsigned int seed) {
  // (0) Create RNG
  std::mt19937 rand(seed);

  if(net_json_.count("nodes") != 1) {
    LOGERROR << "No nodes in configuration!";
    return false;
  }
  JSON nodes_json = net_json_["nodes"];

  // (1) Process input and output nodes
  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;
  NetGraphNode* dataset_input_node = graph.GetInputNodes()[0];

  if(net_json_.count("input") == 1) {
    if(dataset_input_node == nullptr) {
      LOGERROR << "Net requested input, but there are no input nodes in net graph!";
      return false;
    }
    if(net_json_["input"].is_string()) {
      std::string s = net_json_["input"];
      input_nodes.push_back(s);
      LOGDEBUG << "Added input node \"" << s << "\"";
    } else if(net_json_["input"].is_array()) {
      for(unsigned int i = 0; i < net_json_["input"].size(); i++) {
        std::string s = net_json_["input"][i];
        input_nodes.push_back(s);
        LOGDEBUG << "Added input node \"" << s << "\"";
      }
    }
  }

  if(net_json_.count("output") == 1) {
    if(net_json_["output"].is_string()) {
      std::string s = net_json_["output"];
      output_nodes.push_back(s);
      LOGDEBUG << "Added output node \"" << s << "\"";
    } else if(net_json_["output"].is_array()) {
      for(unsigned int i = 0; i < net_json_["output"].size(); i++) {
        std::string s = net_json_["output"][i];
        output_nodes.push_back(s);
        LOGDEBUG << "Added output node \"" << s << "\"";
      }
    }
  }

  // (2) Add layers
  while(true) {
    bool inserted_a_node = false;
    bool should_have_inserted_a_node = false;
    std::string suspect_node = "";

    for(JSON::iterator node_json_iterator = nodes_json.begin(); node_json_iterator != nodes_json.end(); ++node_json_iterator) {
      JSON node_json = node_json_iterator.value();

      bool can_insert_node = true;

      // Check if node exists already
      if(graph.ContainsNode(node_json_iterator.key()))
        continue;
      else
        should_have_inserted_a_node = true;

      std::vector<std::string> node_input_connections;
      // Check if input nodes exist
      // TODO parse buffer ids
      if(node_json["input"].is_string()) {
        std::string s = node_json["input"];

        bool graph_contains_node = graph.ContainsNode(s);
        can_insert_node &= graph_contains_node;

        if(!graph_contains_node && nodes_json.count(s) != 1)
          suspect_node = node_json_iterator.key();

        node_input_connections.push_back(s);
      } else if(node_json["input"].is_array()) {
        for (unsigned int i = 0; i < node_json["input"].size();) {
          std::string s = node_json["input"][i];

          bool graph_contains_node = graph.ContainsNode(s);
          can_insert_node &= graph_contains_node;

          if(!graph_contains_node && nodes_json.count(s) != 1)
            suspect_node = node_json_iterator.key();

          node_input_connections.push_back(s);
        }
      }

      if(can_insert_node) {
        LOGDEBUG << "Inserting node: \"" << node_json_iterator.key() << "\"";

        // Get random seed
        unsigned int new_random_seed = rand();
        node_json = LayerFactory::InjectSeed(node_json, new_random_seed);
        LOGDEBUG << "Inserting " << node_json.dump();

        // Insert YOLO configuration if needed
        if(LayerFactory::ExtractLayerType(node_json).compare("yolo_detection") == 0 ||
          LayerFactory::ExtractLayerType(node_json).compare("yolo_output") == 0 ) {
          node_json["layer"]["yolo_configuration"] = net_json_["yolo_configuration"];
        }

        // Assemble node
        NetGraphNode *node;
        if(LayerFactory::ExtractLayerType(node_json).compare("yolo_output") == 0) {
          Layer* layer = new YOLODynamicOutputLayer(LayerFactory::ExtractConfiguration(node_json), class_manager);
          node = new NetGraphNode(layer);
        } else {
          node = new NetGraphNode(node_json);
        }
        node->unique_name = node_json_iterator.key();

        // Add input connections
        for (std::string &input_connection : node_input_connections) {
          // TODO parse buffer ids
          LOGDEBUG << "  with input: \"" << input_connection << "\"";
          NetGraphNode *source_node = graph.GetNode(input_connection);
          if(source_node == nullptr) {
            LOGERROR << "Node \"" << input_connection << "\" somehow missing from net graph!";
            return false;
          }
          NetGraphConnection source = NetGraphConnection(source_node, 0, true);
          node->input_connections.push_back(source);
        }

        // Add dataset input connection if requested
        for (std::string &input_node_name : input_nodes) {
          if (input_node_name.compare(node_json_iterator.key()) == 0) {
            node->input_connections.push_back(NetGraphConnection(dataset_input_node, 0, false));
            LOGDEBUG << "  with dataset input";
          }
        }

        // Set output flag for output nodes
        for(std::string& output_node_name : output_nodes)
          if(output_node_name.compare(node_json_iterator.key()) == 0)
            node->is_output = true;

        // Insert node
        graph.AddNode(node);
        inserted_a_node = true;
      }
    }
    if(!inserted_a_node && !should_have_inserted_a_node)
      break;
    else if(!inserted_a_node && should_have_inserted_a_node) {
      LOGERROR << "Net configuration invalid, node \"" << suspect_node << "\" likely has wrong inputs";
      return false;
    }
  }

  // (3) Add loss layers to outputs
  for(std::string& output_node_name : output_nodes) {
    NetGraphNode* output_node = graph.GetNode(output_node_name);
    if(output_node == nullptr) {
      LOGERROR << "Net graph does not contain output node \"" << output_node_name << "\"!";
      return false;
    }
    if(net_json_.count("error_layer") == 1 && net_json_["error_layer"].is_string()) {
      std::string error_layer = net_json_["error_layer"];
      if(error_layer.compare("no") == 0) {
        LOGDEBUG << "Not using an error layer";
      } else if (error_layer.compare("dummy") == 0) {
        DummyErrorLayer *error_layer = new DummyErrorLayer();
        NetGraphNode *error_node = new NetGraphNode(error_layer, NetGraphConnection(output_node, 0, true));
        error_node->input_connections.push_back(NetGraphConnection(dataset_input_node, 1, false));
        error_node->input_connections.push_back(NetGraphConnection(dataset_input_node, 3, false));
        error_node->unique_name = "loss_" + output_node_name;
        graph.AddNode(error_node);
      } else if (error_layer.compare("square") == 0) {
        ErrorLayer *error_layer = new ErrorLayer();
        NetGraphNode *error_node = new NetGraphNode(error_layer, NetGraphConnection(output_node, 0, true));
        error_node->input_connections.push_back(NetGraphConnection(dataset_input_node, 1, false));
        error_node->input_connections.push_back(NetGraphConnection(dataset_input_node, 3, false));
        error_node->unique_name = "loss_" + output_node_name;
        graph.AddNode(error_node);
      } else if (error_layer.compare("yolo") == 0) {
        if(net_json_.count("yolo_configuration") != 1 || !net_json_["yolo_configuration"].is_object()) {
          LOGERROR << "Missing YOLO configuration!";
          return false;
        }
        JSON yolo_configuration = net_json_["yolo_configuration"];
        YOLOLossLayer *error_layer = new YOLOLossLayer(yolo_configuration);
        NetGraphNode *error_node = new NetGraphNode(error_layer, NetGraphConnection(output_node, 0, true));
        error_node->input_connections.push_back(NetGraphConnection(dataset_input_node, 1, false));
        error_node->input_connections.push_back(NetGraphConnection(dataset_input_node, 3, false));
        error_node->unique_name = "loss_" + output_node_name;
        graph.AddNode(error_node);
      } else if (error_layer.compare("square") == 0) {
      } else {
        LOGERROR << "Unknown error layer type specified: " << error_layer;
        return false;
      }
    } else {
      LOGERROR << "No error layer type specified!";
      return false;
    }
  }

  graph.Initialize();
  return graph.IsComplete();
}

}
