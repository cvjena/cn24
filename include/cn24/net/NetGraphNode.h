/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file NetGraphNode.h
 * @class NetGraphNode
 * @brief Represents a node in the NetGraph
 * 
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_NETGRAPHNODE_H
#define CONV_NETGRAPHNODE_H

#include "Layer.h"
#include "../util/CombinedTensor.h"
#include "NetStatus.h"
#include "../util/TensorViewer.h"
#include "../net/LayerFactory.h"

#include "StatLayer.h"
#include "NetGraph.h"

#include <vector>
#include <string>

namespace Conv {

class NetGraphNode {
public:
	explicit NetGraphNode(Layer* layer) : layer(layer) {
		layer->CreateBufferDescriptors(output_buffers);
	}

	NetGraphNode(Layer* layer, NetGraphConnection first_connection) : layer(layer) {
		input_connections.push_back(first_connection);
		layer->CreateBufferDescriptors(output_buffers);
	}

	explicit NetGraphNode(JSON descriptor) {
		layer = LayerFactory::ConstructLayer(descriptor);
		if(layer == nullptr) {
			FATAL("Unknown layer! Descriptor: " << descriptor.dump());
		}
		layer->CreateBufferDescriptors(output_buffers);
	}

  NetGraphNode(JSON descriptor, NetGraphConnection first_connection) {
		input_connections.push_back(first_connection);
		layer = LayerFactory::ConstructLayer(descriptor);
		layer->CreateBufferDescriptors(output_buffers);
  }

	Layer* layer;
	std::vector<NetGraphConnection> input_connections;
	std::vector<NetGraphBackpropConnection> backprop_connections;
	std::vector<NetGraphBuffer> output_buffers;

	//int unique_id = -1;
  std::string unique_name = "";
	bool is_output = false;
	bool is_input = false;

	// Status
	bool initialized = false;

	// Flags used by NetGraph functions
	bool flag_ff_visited = false;
	bool flag_bp_visited = false;
};
  
}
#endif