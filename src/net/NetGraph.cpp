/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Log.h"

#include "NetGraph.h"

namespace Conv {

void NetGraph::AddNode(NetGraphNode* node) {
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

}