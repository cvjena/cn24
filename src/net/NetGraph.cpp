/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <sstream>
#include "Log.h"

#include "NetGraph.h"

namespace Conv {

void NetGraph::AddNode(NetGraphNode* node) {
	if (node == nullptr) {
		LOGERROR << "Tried to add null-pointer node!";
		return;
	}
	node->unique_id = ++last_uid;
	nodes_.push_back(node);
}

bool NetGraph::IsComplete() {
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
			if (node->unique_id == -1) {
				LOGWARN << "Node has no unique identifier!";
				is_complete = false;
			}
		}
		else {
			LOGWARN << "Null-pointer node encountered!";
			node_okay = false;
		}

		if (node_okay) {
			LOGINFO << "Node is okay: " << node->layer->GetLayerDescription();
		}
		else {
			LOGINFO << "Node is not okay: " << node->layer->GetLayerDescription();
			is_complete = false;
		}
	}

	LOGINFO << "Graph check complete.";
	return is_complete;
}

void NetGraph::PrintGraph(std::ostream& graph_output) {
	std::ostringstream node_output;
	std::ostringstream edge_output;

	for (NetGraphNode* node : nodes_) {
		// 1. Print node details
		node_output << "node" << node->unique_id << " [shape=record, label=\""
			<< "{" << node->layer->GetLayerDescription();
		if (node->output_buffers.size() > 1) {
			node_output << "| {";
			for (unsigned int i = 0; i < node->output_buffers.size(); i++) {
				if (i > 0)
					node_output << "|";
				node_output << "<o" << i << ">" << node->output_buffers[i].description;
			}
			node_output << "}";
		}
		else if (node->output_buffers.size() == 1) {
			node_output << "| <o0> " << node->output_buffers[0].description;
		}
		node_output << "}\"];\n";

		// 2. Print edges
		for (NetGraphConnection connection : node->input_connections) {
			edge_output << "node" << connection.node->unique_id << ":o" << connection.buffer << " -> node"
				<< node->unique_id << ";\n";
		}
	}

	graph_output << node_output.str();
	graph_output << edge_output.str();
}

}