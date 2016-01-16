/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <sstream>
#include <algorithm>

#include "Log.h"
#include "LossFunctionLayer.h"
#include "TrainingLayer.h"
#include "GradientAccumulationLayer.h"
#include "StatLayer.h"

#include "NetGraph.h"

#include "TensorViewer.h"

namespace Conv {

void NetGraph::AddNode(NetGraphNode* node) {
	// Validate node
	if (node == nullptr)
		FATAL("Tried to add null-pointer node!");

	if (node->layer == nullptr)
		FATAL("Tried to add layerless node!");

	// We don't check the node's connections at this time because we can't expect
	// the user to sort the nodes before adding. NetGraph::IsComplete() contains
	// all necessary checks.

	// Assign the node a new unique ID if it doesn't have one already
  if (node->unique_name.length() == 0) {
    int node_id = ++last_uid;
    std::stringstream ss; ss << "node" << node_id;
    node->unique_name = ss.str();
  }

	// Add node to list
	nodes_.push_back(node);

	// Add node to registries
	if (node->is_input)
		input_nodes_.push_back(node);
	if (node->is_output)
		output_nodes_.push_back(node);

	if (dynamic_cast<StatLayer*>(node->layer) != NULL)
		stat_nodes_.push_back(node);
	if (dynamic_cast<LossFunctionLayer*>(node->layer) != NULL)
		loss_nodes_.push_back(node);
	if (dynamic_cast<TrainingLayer*>(node->layer) != NULL)
		training_nodes_.push_back(node);
  
  // Add backprop connection where appropiate
  for(NetGraphConnection& connection : node->input_connections) {
    if (connection.backprop && !connection.node->is_input) {
      NetGraphBackpropConnection backprop_connection(node, connection.buffer);
      connection.node->backprop_connections.push_back(backprop_connection);
    }
    else if (connection.backprop && connection.node->is_input) {
      connection.backprop = false;
    }
  }
}

bool NetGraph::IsComplete() const {
	unsigned int inputs = 0, outputs = 0;
	bool is_complete = true;

	for (NetGraphNode* node : nodes_) {
		bool node_okay = true;
		if (node != nullptr) {
			if (node->layer != nullptr) {
				for (NetGraphConnection connection : node->input_connections) {
					if (connection.node != nullptr) {
						if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
							LOGWARN << "Node has out-of-network connection: " << node->layer->GetLayerDescription();
						}
						else {
							if (connection.buffer >= connection.node->output_buffers.size()) {
								LOGWARN << "Node's connection points to invalid buffer: " << node->layer->GetLayerDescription();
								node_okay = false;
							}
						}
					}
					else {
						LOGWARN << "Node has null-pointer connection: " << node->layer->GetLayerDescription();
						node_okay = false;
					}
				}
			}
			else {
				LOGWARN << "Node has null-pointer for layer!";
				node_okay = false;
			}
			if (node->unique_name.length() == 0) {
				LOGWARN << "Node has no unique identifier!";
				node_okay = false;
			}
			if (node->is_input && node->is_output) {
				LOGWARN << "Node is both input and output!";
				node_okay = false;
			}
			if (node->is_input)
				inputs++;
			if (node->is_output)
				outputs++;
		}
		else {
			LOGWARN << "Null-pointer node encountered!";
			node_okay = false;
		}

		if (node_okay) {
			LOGDEBUG << "Node is okay: " << node->layer->GetLayerDescription();
		}
		else {
			LOGWARN << "Node is not okay: " << node->layer->GetLayerDescription();
			is_complete = false;
		}
	}

	if (inputs == 0) {
		LOGWARN << "Net has no inputs!";
		is_complete = false;
	}

	if (outputs == 0) {
		LOGWARN << "Net has no outputs!";
		is_complete = false;
	}

	LOGDEBUG << "Graph check complete.";
	return is_complete;
}

void NetGraph::PrintGraph(std::ostream& graph_output) {
	if (nodes_.size() == 0)
		return;

	std::ostringstream node_output;
	std::ostringstream edge_output;

	node_output << "graph [ranksep=.75, esep=1];";

	for (NetGraphNode* node : nodes_) {

		// 1. Print node details
		node_output << node->unique_name << " [shape=record,";

		if (node->is_input) {
			node_output << "color=red,";
		}
		if (node->is_output) {
			node_output << "color=blue,";
		}
			
		node_output << " label=\""
			<< "{ <i> " << node->unique_name << ": " << node->layer->GetLayerDescription();
		if (node->output_buffers.size() > 1) {
			node_output << "| {";
			for (unsigned int i = 0; i < node->output_buffers.size(); i++) {
				if (i > 0)
					node_output << "|";
				node_output << "<o" << i << ">" << node->output_buffers[i].description << " " << node->output_buffers[i].combined_tensor->data;
			}
			node_output << "}";
		}
		else if (node->output_buffers.size() == 1) {
			node_output << "| <o0> " << node->output_buffers[0].description << " " << node->output_buffers[0].combined_tensor->data;
		}
		node_output << "}\"];\n";

		// 2. Print edges
		for (NetGraphConnection connection : node->input_connections) {
			edge_output << connection.node->unique_name << ":o" << connection.buffer << " -> "
				<< node->unique_name << ":i" <<
				"[penwidth=2";
			if (!connection.backprop)
				edge_output << ",style=dotted";
			edge_output << "];\n";
		}

		/*for (NetGraphBackpropConnection backprop_connection : node->backprop_connections) {
			edge_output << "node" << backprop_connection.node->unique_id << ":i -> node"
				<< node->unique_id << ":o" << backprop_connection.buffer <<
				"[penwidth=5,style=dotted,arrowsize=.6];\n";
		}*/
	}

	graph_output << node_output.str();
	graph_output << edge_output.str();
}

void NetGraph::Initialize() {
	// check for nodes with multiple backprop connections
  bool no_multiple_connections = true;
  do {
    no_multiple_connections = true;
		std::vector<NetGraphNode*> nodes(nodes_);
    for (NetGraphNode* node : nodes) {
      if(node->backprop_connections.size() > 1 && dynamic_cast<GradientAccumulationLayer*>(node->layer) == NULL) {
        no_multiple_connections = false;
        LOGINFO << "Node has multiple backprop connections: " << node->layer->GetLayerDescription();
        
        // Insert gradient accumulation layer
        GradientAccumulationLayer* ga = new GradientAccumulationLayer(node->backprop_connections.size());
        NetGraphNode* ga_node = new NetGraphNode(ga, NetGraphConnection(node));
        AddNode(ga_node);
        
        // Redirect input connections using backprop connections
        int b = 0;
        for(NetGraphBackpropConnection& backprop_connection : node->backprop_connections) {
          if(backprop_connection.node != ga_node) {
            for(NetGraphConnection& target_connection : backprop_connection.node->input_connections) {
              if(target_connection.node == node && target_connection.buffer == backprop_connection.buffer && target_connection.backprop) {
                target_connection.node = ga_node;
                backprop_connection.buffer = b;
                target_connection.buffer = b++;
              }
            }
            ga_node->backprop_connections.push_back(backprop_connection);
          }
        }
        
        // Remove backprop connections from node
        auto predicate = 
          [&](NetGraphBackpropConnection backprop_connection){ return backprop_connection.node != ga_node; };
        node->backprop_connections.erase(std::remove_if(node->backprop_connections.begin(), node->backprop_connections.end(),
        predicate), node->backprop_connections.end());
      }
    }
  } while(!no_multiple_connections);
  
  for (NetGraphNode* node : nodes_){
		InitializeNode(node);
	}

}

void NetGraph::InitializeNode(NetGraphNode* node) {
	if (!node->initialized) {
		// Collect input tensors through DFS
		std::vector<CombinedTensor*> input_tensors;
		for (NetGraphConnection connection : node->input_connections) {
			InitializeNode(connection.node);


			input_tensors.push_back(connection.node->output_buffers[connection.buffer].combined_tensor);
		}

		// Ask layer to create output buffers
		std::vector<CombinedTensor*> output_tensors;
		bool success_outputs = node->layer->CreateOutputs(input_tensors, output_tensors);

		// Verify output buffer creation
		if (!success_outputs) {
			FATAL("Layer will not create outputs: " << node->layer->GetLayerDescription() << ", input0: " << input_tensors[0]->data);
    }

		// Verify output buffer count
		if (output_tensors.size() != node->output_buffers.size())
			FATAL("Node created wrong number of output buffers!");

		// Update node output buffer info
		for (unsigned int b = 0; b < output_tensors.size(); b++) {
			node->output_buffers[b].combined_tensor = output_tensors[b];
		}

		// Connect layer
		bool success_connect = node->layer->Connect(input_tensors, output_tensors, this);
		if (!success_connect)
			FATAL("Layer will not connect: " << node->layer->GetLayerDescription());

		// Save to flag
		node->initialized = true;
	}
}

void NetGraph::FeedForward() {
	FeedForward(nodes_, true);
}

void NetGraph::FeedForward(std::vector<NetGraphNode*>& nodes, bool clear_flag) {
	if (clear_flag)
		for (NetGraphNode* node : nodes)
			node->flag_ff_visited = false;

	for (NetGraphNode* node : nodes)
		FeedForward(node);
}

void NetGraph::FeedForward(NetGraphNode* node) {
	if (!node->flag_ff_visited) {
		// Make sure all input nodes have valid outputs
		for (NetGraphConnection connection : node->input_connections)
			FeedForward(connection.node);

#ifdef LAYERTIME
    auto t_begin = std::chrono::system_clock::now();
#endif

		PrepareNode(node);
		// Call the Layer::FeedForward method and set the visited flag
		node->layer->FeedForward();
    if(layerview_enabled_)
      for(NetGraphBuffer buffer: node->output_buffers) {
        for(unsigned int sample = 0; sample < buffer.combined_tensor->data.samples(); sample++) {
          for(unsigned int map = 0; map < buffer.combined_tensor->data.maps(); map++) {
            std::stringstream ss;
            ss << node->unique_name << ": " << node->layer->GetLayerDescription() << ", buffer " << buffer.description;
  #ifdef BUILD_OPENCL
            buffer.combined_tensor->data.MoveToCPU();
  #endif
            viewer.show(&(buffer.combined_tensor->data), ss.str(), false, map, sample);
          }
        }
      }
    
		node->flag_ff_visited = true;

#ifdef LAYERTIME
    auto t_end = std::chrono::system_clock::now();
    std::chrono::duration<double> pass_duration = t_end - t_begin;
    LOGINFO << "FeedFwd Layer " << node->unique_name << " (" << node->layer->GetLayerDescription() << ") time:\t" << pass_duration.count() << "s";
#endif
	}
}

void NetGraph::BackPropagate() {
	BackPropagate(nodes_, true);
}

void NetGraph::BackPropagate(std::vector<NetGraphNode*>& nodes, bool clear_flag) {
	if (clear_flag)
		for (NetGraphNode* node : nodes)
			node->flag_bp_visited = false;

	for (NetGraphNode* node : nodes)
		BackPropagate(node);
}

void NetGraph::BackPropagate(NetGraphNode* node) {
	if (!node->flag_bp_visited) {
		// Make sure all source nodes have valid gradients
		for (NetGraphBackpropConnection backprop_connection : node->backprop_connections)
			BackPropagate(backprop_connection.node);

		bool do_backprop = false;
		for (NetGraphConnection connection : node->input_connections)
			do_backprop |= connection.backprop;

#ifdef LAYERTIME
    auto t_begin = std::chrono::system_clock::now();
#endif

		PrepareNode(node);
		node->layer->SetBackpropagationEnabled(do_backprop);
		// Call the Layer::FeedForward method and set the visited flag
		node->layer->BackPropagate();
		node->flag_bp_visited = true;

#ifdef LAYERTIME
    auto t_end = std::chrono::system_clock::now();
    std::chrono::duration<double> pass_duration = t_end - t_begin;
    LOGINFO << "BackProp Layer " << node->unique_name << " (" << node->layer->GetLayerDescription() << ") time:\t" << pass_duration.count() << "s";
#endif
	}
}

void NetGraph::GetParameters (std::vector< CombinedTensor* >& parameters) {
	for (NetGraphNode* node : nodes_) {
		Layer* layer = node->layer;
    for (unsigned int p = 0; p < layer->parameters().size(); p++) {
      parameters.push_back (layer->parameters() [p]);
    }
  }
}

void NetGraph::SerializeParameters(std::ostream& output) {
	// TODO use unique layer ids
	for (unsigned int l = 0; l < nodes_.size(); l++) {
		Layer* layer = nodes_[l]->layer;
		for (unsigned int p = 0; p < layer->parameters().size(); p++) {
			layer->parameters()[p]->data.Serialize(output);
		}
	}
}

void NetGraph::DeserializeParameters(std::istream& input, unsigned int last_layer) {
	// TODO use unique layer ids
  if (last_layer == 0 || last_layer >= nodes_.size())
    last_layer = nodes_.size() - 1;
  for (unsigned int l = 0; l <= last_layer; l++) {
    Layer* layer = nodes_[l]->layer;
    for (unsigned int p = 0; p < layer->parameters().size(); p++) {
      if (!input.good() || input.eof())
        break;
      unsigned int elements_before = layer->parameters() [p]->data.elements();
      layer->parameters() [p]->data.Deserialize (input);
      unsigned int elements_after = layer->parameters() [p]->data.elements();
      LOGINFO << "Loaded parameters for layer " << l << " parameter set " << p << ": " << layer->parameters()[p]->data;
      if(elements_before != elements_after) {
        LOGERROR << "Deserialization changed layer parameter count!";
      }
      input.peek();
    }
  }
}

void NetGraph::InitializeWeights() {
	for (NetGraphNode* node : nodes_)
		node->flag_bp_visited = false;

	for (NetGraphNode* node : nodes_)
		InitializeWeights(node);

	for (NetGraphNode* node : nodes_)
		node->flag_bp_visited = false;
}

void NetGraph::InitializeWeights(NetGraphNode* node) {
	if (!node->flag_bp_visited) {
		std::vector<Layer*> layers_to_connect;
		for (NetGraphBackpropConnection backprop_connection : node->backprop_connections) {
			InitializeWeights(backprop_connection.node);
			layers_to_connect.push_back(backprop_connection.node->layer);
		}
		node->layer->OnLayerConnect(layers_to_connect);

		node->flag_bp_visited = true;
	}
}

void NetGraph::PrepareNode(NetGraphNode* node) {
#ifdef BUILD_OPENCL
	if (!node->layer->IsOpenCLAware()) {
		for (NetGraphConnection connection : node->input_connections) {
			connection.node->output_buffers[connection.buffer].combined_tensor->data.MoveToCPU();
			connection.node->output_buffers[connection.buffer].combined_tensor->delta.MoveToCPU();
		}
		for (NetGraphBuffer& buffer : node->output_buffers) {
			buffer.combined_tensor->data.MoveToCPU();
			buffer.combined_tensor->delta.MoveToCPU();
		}
	}
#endif
}

datum NetGraph::AggregateLoss() {
	datum loss = 0;
	for (NetGraphNode* loss_node : GetLossNodes()) {
		LossFunctionLayer* loss_layer = dynamic_cast<LossFunctionLayer*>(loss_node->layer);
		if (loss_layer != NULL) {
			loss += loss_layer->CalculateLossFunction();
		} else {
			FATAL("Null pointer in loss node encountered!");
		}
	}
	return loss;
}

}