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
#include "NetGraphNode.h"

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
  
void NetGraph::SetStatLayersEnabled(bool enabled) {
  for (unsigned int n = 0; n < GetStatNodes().size(); n++) {
    StatLayer* stat_layer = dynamic_cast<StatLayer*>(GetStatNodes()[n]->layer);
    stat_layer->SetDisabled(!enabled);
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
				if(node->output_buffers[i].combined_tensor->metadata != nullptr)
					node_output << " with metadata";
				if(node->output_buffers[i].combined_tensor->is_dynamic)
					node_output << " (dynamic)";
			}
			node_output << "}";
		}
		else if (node->output_buffers.size() == 1) {
			node_output << "| <o0> " << node->output_buffers[0].description << " " << node->output_buffers[0].combined_tensor->data;
			if(node->output_buffers[0].combined_tensor->metadata != nullptr)
				node_output << " with metadata";
			if(node->output_buffers[0].combined_tensor->is_dynamic)
				node_output << " (dynamic)";
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
    bool layer_has_dynamic_inputs = false;
		// Collect input tensors through DFS
		std::vector<CombinedTensor*> input_tensors;
		for (NetGraphConnection connection : node->input_connections) {
			InitializeNode(connection.node);
			input_tensors.push_back(connection.node->output_buffers[connection.buffer].combined_tensor);
			layer_has_dynamic_inputs |= connection.node->output_buffers[connection.buffer].combined_tensor->is_dynamic;
		}
    // See if layer is dynamic tensor aware
		if(layer_has_dynamic_inputs) {
			if(!node->layer->IsDynamicTensorAware()) {
				FATAL("Layer has dynamic input but doesn't support it: " << node->layer->GetLayerDescription());
			}
		}

		// Ask layer to create output buffers
		std::vector<CombinedTensor*> output_tensors;
		bool success_outputs = node->layer->CreateOutputs(input_tensors, output_tensors);

		// Verify output buffer creation
		if (!success_outputs) {
			FATAL("Layer will not create outputs: " << node->layer->GetLayerDescription() << ", inputs: " << input_tensors.size());
    }

		// Verify output buffer count
		if (output_tensors.size() != node->output_buffers.size())
			FATAL("Node \"" << node->unique_name <<  "\" created wrong number of output buffers! (" << output_tensors.size() << " instead of " << node->output_buffers.size() << ")");

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
  OnBeforeFeedForward();
	FeedForward(nodes_, true);
	OnAfterFeedForward();
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

		PrepareNode(node);
#ifdef LAYERTIME
    auto t_begin = std::chrono::system_clock::now();
#endif

		// Call the Layer::FeedForward method and set the visited flag
		node->layer->FeedForward();
    if(layerview_enabled_)
      for(NetGraphBuffer buffer: node->output_buffers) {
				for(unsigned int sample = 0; sample < buffer.combined_tensor->data.samples() && sample < 4; sample++) {
					for(unsigned int map = 0; map < 1; map++) {
        //for(unsigned int sample = 0; sample < buffer.combined_tensor->data.samples(); sample++) {
        //  for(unsigned int map = 0; map < buffer.combined_tensor->data.maps(); map++) {
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
	OnBeforeBackPropagate();
	BackPropagate(nodes_, true);
	OnAfterBackPropagate();
}

void NetGraph::BackPropagate(std::vector<NetGraphNode*>& nodes, bool clear_flag) {
	if (clear_flag)
		for (NetGraphNode* node : nodes)
			node->flag_bp_visited = false;

	for (NetGraphNode* node : nodes) {
		// Check if layer actually needs backprop
		if(node->layer->local_lr_ > 0 && node->layer->parameters_.size() > 0)
      BackPropagate(node);
	}
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
	uint64_t magic = CN24_PAREX_MAGIC;
	output.write((char*)&magic, sizeof(uint64_t)/sizeof(char));
    
	for (unsigned int l = 0; l < nodes_.size(); l++) {
    NetGraphNode* node = nodes_[l];
    Layer* layer = nodes_[l]->layer;
    unsigned int layer_parameters = layer->parameters().size();
    if(layer_parameters > 0) {
      // Write length of node name
      unsigned int node_unique_name_length = node->unique_name.length();
			unsigned int parameter_set_size = node->layer->parameters().size();
      output.write((const char*)&node_unique_name_length, sizeof(unsigned int)/sizeof(char));
      // Write node name
      output.write(node->unique_name.c_str(), node_unique_name_length);

      if (layer->IsSerializationAware()) {
        bool success = layer->Serialize(output);
        if (!success) {
          LOGERROR << "Could not serialize " << node->unique_name << ", lost the stream.";
          return;
        }
      } else {
        unsigned int metadata_length = 0;

        

        output.write((const char*)&metadata_length, sizeof(unsigned int) / sizeof(char));
        output.write((const char*)&parameter_set_size, sizeof(unsigned int) / sizeof(char));

        // Write parameters
        for (unsigned int p = 0; p < layer->parameters().size(); p++) {
          layer->parameters()[p]->data.Serialize(output);
        }
      }
    }
	}
}

void NetGraph::DeserializeParameters(std::istream& input) {
  // Check for right magic number
	uint64_t magic = 0;
	input.read((char*)&magic, sizeof(uint64_t)/sizeof(char));

	if(magic == CN24_PAR_MAGIC) {
		while (input.good() && !input.eof()) {
			// Read node name length
			unsigned int node_unique_name_length;
			unsigned int parameter_set_size;
			input.read((char *) &node_unique_name_length, sizeof(unsigned int) / sizeof(char));
			input.read((char *) &parameter_set_size, sizeof(unsigned int) / sizeof(char));

			// Read node name
			char *node_name_cstr = new char[node_unique_name_length + 1];

			input.read(node_name_cstr, node_unique_name_length);
			node_name_cstr[node_unique_name_length] = '\0';

			std::string node_name(node_name_cstr);

			// Find node
			bool found_node = false;
			for (NetGraphNode *node : nodes_) {
				if (node->unique_name.compare(node_name) == 0) {
					if (node->layer->parameters().size() != parameter_set_size) {
						LOGERROR << "Node name matches, but parameter set size does not";
						continue;
					}
					found_node = true;
					Layer *layer = node->layer;

					// Read parameters
					for (unsigned int p = 0; p < layer->parameters().size(); p++) {
						unsigned int elements_before = layer->parameters()[p]->data.elements();
						layer->parameters()[p]->data.Deserialize(input);
						unsigned int elements_after = layer->parameters()[p]->data.elements();
						LOGDEBUG << "Loaded parameters for node \"" << node->unique_name << "\" parameter set " << p << ": " <<
										 layer->parameters()[p]->data;
						if (elements_before != elements_after) {
							LOGERROR << "Deserialization changed layer parameter count!";
						}
					}

					// Update EOF flag
					input.peek();
					break;
				}
			}
			if (!found_node) {
				LOGWARN << "Could not find node \"" << node_name << "\"";
				for (unsigned int p = 0; p < parameter_set_size; p++) {
					Tensor *t = new Tensor();
					t->Deserialize(input);
					LOGDEBUG << "Skipping parameter set for node " << node_name << *t;
					delete t;
				}
				input.peek();
			}

			delete[] node_name_cstr;
		}
	} else if(magic == CN24_PAREX_MAGIC) {
    // New extended format
		while (input.good() && !input.eof()) {
			// Read node name length
			unsigned int node_unique_name_length;
			unsigned int metadata_length;
			unsigned int parameter_set_size;

			input.read((char *) &node_unique_name_length, sizeof(unsigned int) / sizeof(char));

			// Read node name
			char *node_name_cstr = new char[node_unique_name_length + 1];
			input.read(node_name_cstr, node_unique_name_length);
			node_name_cstr[node_unique_name_length] = '\0';
			std::string node_name(node_name_cstr);


			// Read metadata
			input.read((char *) &metadata_length, sizeof(unsigned int) / sizeof(char));
			char *metadata_cstr = new char[metadata_length + 1];
			if(metadata_length > 0) {
				input.read(metadata_cstr, metadata_length);
			}
			metadata_cstr[metadata_length] = '\0';

			input.read((char *) &parameter_set_size, sizeof(unsigned int) / sizeof(char));
			// Find node
			bool found_node = false;
			for (NetGraphNode *node : nodes_) {
				if (node->unique_name.compare(node_name) == 0) {
					if (node->layer->parameters().size() != parameter_set_size) {
						LOGERROR << "Node name matches, but parameter set size does not";
						continue;
					}
					found_node = true;
					Layer *layer = node->layer;

          if(layer->IsSerializationAware() > 0) {
						// Use new extended deserializer
						bool success = layer->Deserialize(metadata_length, metadata_cstr, parameter_set_size, input);
						if(!success) {
							LOGERROR << "Error when deserializing " << node_name << ", lost the stream";
							return;
						}
					} else {
						// Read parameters
						for (unsigned int p = 0; p < layer->parameters().size(); p++) {
							unsigned int elements_before = layer->parameters()[p]->data.elements();
							layer->parameters()[p]->data.Deserialize(input);
							unsigned int elements_after = layer->parameters()[p]->data.elements();
							LOGDEBUG << "Loaded parameters for node \"" << node->unique_name << "\" parameter set " << p << ": " <<
											 layer->parameters()[p]->data;
							if (elements_before != elements_after) {
								LOGERROR << "Deserialization changed layer parameter count!";
							}
						}
					}

					// Update EOF flag
					input.peek();
					break;
				}
			}
			if (!found_node) {
				if(node_name.compare(0, 2, "__") == 0) {
					// Skip this info
					LOGDEBUG << "Skipping metadata segment " << node_name;
				} else {
					LOGWARN << "Could not find node \"" << node_name << "\"";
				}
				for (unsigned int p = 0; p < parameter_set_size; p++) {
					Tensor *t = new Tensor();
					t->Deserialize(input);
					LOGDEBUG << "Skipping parameter set for node " << node_name << *t;
					delete t;
				}
				input.peek();
			}

			delete[] node_name_cstr;
		}
	} else {
    FATAL("Wrong magic at start of stream!");
  }

}

void NetGraph::InitializeWeights(bool no_init) {
	for (NetGraphNode* node : nodes_)
		node->flag_bp_visited = false;

	for (NetGraphNode* node : nodes_)
		InitializeWeights(node, no_init);

	for (NetGraphNode* node : nodes_)
		node->flag_bp_visited = false;
}

void NetGraph::InitializeWeights(NetGraphNode* node, bool no_init) {
	if (!node->flag_bp_visited) {
		std::vector<Layer*> layers_to_connect;
		for (NetGraphBackpropConnection backprop_connection : node->backprop_connections) {
			InitializeWeights(backprop_connection.node, no_init);
			layers_to_connect.push_back(backprop_connection.node->layer);
		}
		node->layer->OnLayerConnect(layers_to_connect, no_init);

		node->flag_bp_visited = true;
	}
}

void NetGraph::PrepareNode(NetGraphNode* node) {
#ifdef BUILD_OPENCL
	if (!node->layer->IsGPUMemoryAware()) {
#ifdef LAYERTIME
		auto t_begin = std::chrono::system_clock::now();
#endif
		for (NetGraphConnection connection : node->input_connections) {
			connection.node->output_buffers[connection.buffer].combined_tensor->data.MoveToCPU();
			connection.node->output_buffers[connection.buffer].combined_tensor->delta.MoveToCPU();
		}
		for (NetGraphBuffer& buffer : node->output_buffers) {
			buffer.combined_tensor->data.MoveToCPU();
			buffer.combined_tensor->delta.MoveToCPU();
		}
#ifdef LAYERTIME
		auto t_end = std::chrono::system_clock::now();
		std::chrono::duration<double> pass_duration = t_end - t_begin;
		LOGINFO << "OpenCL x-fer  " << node->unique_name << " (" << node->layer->GetLayerDescription() << ") time:\t" << pass_duration.count() << "s";
#endif
	}
#else
  UNREFERENCED_PARAMETER(node);
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

bool NetGraph::ContainsNode(const std::string &unique_name) {
	for(NetGraphNode* node: nodes_) {
		if(unique_name.compare(node->unique_name) == 0)
			return true;
	}
	return false;
}

NetGraphNode* NetGraph::GetNode(const std::string &unique_name) {
	for(NetGraphNode* node: nodes_) {
		if(unique_name.compare(node->unique_name) == 0)
			return node;
	}
	return nullptr;
}

void NetGraph::OnBeforeFeedForward() {
	for(unsigned int h=0; h < handler_before_ff_.size(); h++) {
		handler_before_ff_[h](this, NETGRAPH_EVENT_BEFORE_FF);
	}
}

void NetGraph::OnAfterFeedForward() {
	for(unsigned int h=0; h < handler_after_ff_.size(); h++) {
		handler_after_ff_[h](this, NETGRAPH_EVENT_AFTER_FF);
	}
}

void NetGraph::OnBeforeBackPropagate() {
	for(unsigned int h=0; h < handler_before_bp_.size(); h++) {
		handler_before_bp_[h](this, NETGRAPH_EVENT_BEFORE_BP);
	}
}

void NetGraph::OnAfterBackPropagate() {
	for(unsigned int h=0; h < handler_after_bp_.size(); h++) {
		handler_after_bp_[h](this, NETGRAPH_EVENT_AFTER_BP);
	}
}
}