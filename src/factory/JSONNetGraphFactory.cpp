/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */


#include <string>
#include <vector>
#include <cn24/net/ErrorLayer.h>

#include "NetGraph.h"
#include "NetGraphNode.h"
#include "LayerFactory.h"

#include "JSONNetGraphFactory.h"

namespace Conv {

bool JSONNetGraphFactory::AddLayers(NetGraph &graph) {
  // TODO Validate
  // TODO Check GetNode() calls for null pointers

  JSON nodes_json = net_json_["nodes"];

  // (1) Process input and output nodes
  std::vector<std::string> input_nodes;
  std::vector<std::string> output_nodes;
  NetGraphNode* input_node = graph.GetTrainingNodes()[0];

  if(net_json_.count("input") == 1) {
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
    for(JSON::iterator node_json_iterator = nodes_json.begin(); node_json_iterator != nodes_json.end(); ++node_json_iterator) {
      JSON node_json = node_json_iterator.value();

      bool can_insert_node = true;

      // Check if node exists already
      if(graph.ContainsNode(node_json_iterator.key()))
        continue;

      std::vector<std::string> node_input_connections;
      // Check if input nodes exist
      // TODO parse buffer ids
      if(node_json["input"].is_string()) {
        std::string s = node_json["input"];
        can_insert_node &= graph.ContainsNode(s);
        node_input_connections.push_back(s);
      } else if(node_json["input"].is_array()) {
        for (unsigned int i = 0; i < node_json["input"].size();) {
          std::string s = node_json["input"][i];
          can_insert_node &= graph.ContainsNode(s);
          node_input_connections.push_back(s);
        }
      }

      if(can_insert_node) {
        LOGDEBUG << "Inserting node: \"" << node_json_iterator.key() << "\"";
        // Assemble node
        NetGraphNode *node = new NetGraphNode(node_json);
        node->unique_name = node_json_iterator.key();

        // Add input connections
        for (std::string &input_connection : node_input_connections) {
          // TODO parse buffer ids
          LOGDEBUG << "  with input: \"" << input_connection << "\"";
          NetGraphNode *source_node = graph.GetNode(input_connection);
          NetGraphConnection source = NetGraphConnection(source_node, 0, true);
          node->input_connections.push_back(source);
        }

        // Add dataset input connection if requested
        for (std::string &input_node_name : input_nodes) {
          if (input_node_name.compare(node_json_iterator.key()) == 0) {
            node->input_connections.push_back(NetGraphConnection(input_node, 0, false));
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
    if(!inserted_a_node)
      break;
  }

  // (3) Add loss layers to outputs
  for(std::string& output_node_name : output_nodes) {
    NetGraphNode* output_node = graph.GetNode(output_node_name);
    ErrorLayer* error_layer = new ErrorLayer();
    NetGraphNode* error_node = new NetGraphNode(error_layer,NetGraphConnection(output_node, 0, true));
    error_node->input_connections.push_back(NetGraphConnection(input_node, 1, false));
    error_node->input_connections.push_back(NetGraphConnection(input_node, 3, false));
    error_node->unique_name = "loss_" + output_node_name;
    graph.AddNode(error_node);
  }

  graph.Initialize();
  return graph.IsComplete();
}

}
