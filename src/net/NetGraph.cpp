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
		if (node != nullptr) {
			if (node->layer != nullptr) {
				for (NetGraphConnection* connection : node->input_connections) {
					if (connection->node != nullptr) {
						if (std::find(nodes_.begin(), nodes_.end(), node) == nodes_.end()) {
							LOGWARN << "Node has out-of-network connection: " << node->layer->GetLayerDescription();
						}
						else {
							if (connection->buffer < node->output_buffers.size()) {
								LOGINFO << "Node is okay: " << node->layer->GetLayerDescription();
							}
							else {
								LOGWARN << "Node's connection points to invalid buffer: " << node->layer->GetLayerDescription();
								is_complete = false;
							}
						}
					}
					else {
						LOGWARN << "Node has null-pointer connection: " << node->layer->GetLayerDescription();
						is_complete = false;
					}
				}
			}
			else {
				LOGWARN << "Node has null-pointer for layer!";
				is_complete = false;
			}
		}
		else {
			LOGWARN << "Null-pointer node encountered!";
			is_complete = false;
		}
	}

	return is_complete;
}

}