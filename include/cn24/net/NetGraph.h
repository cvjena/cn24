/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file NetGraph.h
 * @class NetGraph
 * @brief Represents a neural network as a directed acyclic graph
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_NETGRAPH_H
#define CONV_NETGRAPH_H

#include "Layer.h"
#include "CombinedTensor.h"

#include <vector>

namespace Conv {

class NetGraphNode;

struct NetGraphConnection {
public:
	NetGraphConnection() {}
	explicit NetGraphConnection(NetGraphNode* node, unsigned int buffer = 0) : node(node), buffer(buffer) {}
	NetGraphNode* node = nullptr;
	unsigned int buffer = 0;
};

class NetGraphBuffer {
public:
	std::string description = "Output";
	CombinedTensor* combined_tensor = nullptr;
};

class NetGraphNode {
public:
	explicit NetGraphNode(Layer* layer) : layer(layer) {
		layer->CreateBufferDescriptors(output_buffers);
	}
	NetGraphNode(Layer* layer, NetGraphConnection first_connection) : layer(layer) {
		input_connections.push_back(first_connection);
		layer->CreateBufferDescriptors(output_buffers);
	}

	Layer* layer;
	std::vector<NetGraphConnection> input_connections;
	std::vector<NetGraphBuffer> output_buffers;

	bool initialized = false;
	int graph_status = 0;
	int unique_id = -1;
};

class NetGraph {
public:
	// Graph manipulation
	void AddNode(NetGraphNode* node);

	// Output
	void PrintGraph(std::ostream& graph_output);

	// Status
	bool IsComplete() const;
private:
	std::vector<NetGraphNode*> nodes_;

	int last_uid = -1;
};

}

#endif
