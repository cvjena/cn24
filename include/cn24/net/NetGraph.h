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

class NetGraphConnection {
public:
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
	Layer* layer;
	std::vector<NetGraphConnection*> input_connections;
	std::vector<NetGraphBuffer*> output_buffers;
};

class NetGraph {
public:
	void AddNode(NetGraphNode* node);
	bool IsComplete();
private:
	std::vector<NetGraphNode*> nodes_;
};

}

#endif
