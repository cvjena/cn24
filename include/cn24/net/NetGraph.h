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
#include "NetStatus.h"

#include <vector>

namespace Conv {

class NetGraphNode;

struct NetGraphConnection {
public:
	NetGraphConnection() {}
	explicit NetGraphConnection(NetGraphNode* node, unsigned int buffer = 0, bool backprop = true) : node(node), buffer(buffer), backprop(backprop) {}
	NetGraphNode* node = nullptr;
	unsigned int buffer = 0;
	bool backprop = true;
};

struct NetGraphBackpropConnection {
	NetGraphBackpropConnection() {}
	NetGraphBackpropConnection(NetGraphNode* node, unsigned int buffer) : node(node), buffer(buffer) {}
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
	std::vector<NetGraphBackpropConnection> backprop_connections;
	std::vector<NetGraphBuffer> output_buffers;

	int unique_id = -1;
	bool is_output = false;
	bool is_input = false;

	// Status
	bool initialized = false;

	// Flags used by NetGraph functions
	bool flag_ff_visited = false;
	bool flag_bp_visited = false;
};

class NetGraph : public NetStatus {
public:
	// Graph manipulation
	void AddNode(NetGraphNode* node);
	void Initialize();

	// Node queries
	inline std::vector<NetGraphNode*>& GetOutputNodes() { return output_nodes_; }
	inline NetGraphNode* GetDefaultOutputNode() { return output_nodes_.size() > 0 ? output_nodes_[0] : nullptr; }

	inline std::vector<NetGraphNode*>& GetStatNodes() { return stat_nodes_; }
	inline std::vector<NetGraphNode*>& GetLossNodes() { return loss_nodes_; }
	inline std::vector<NetGraphNode*>& GetTrainingNodes() { return training_nodes_; }
	inline std::vector<NetGraphNode*>& GetNodes() { return nodes_; }

	// Network 
	void FeedForward();
	void FeedForward(std::vector<NetGraphNode*>& nodes, bool clear_flag = true);
	void BackPropagate();
	void BackPropagate(std::vector<NetGraphNode*>& nodes, bool clear_flag = true);

	// Parameter management
  void InitializeWeights();
  void GetParameters(std::vector<CombinedTensor*>& parameters);
  void SerializeParameters(std::ostream& output);
  void DeserializeParameters(std::istream& input, unsigned int last_layer = 0);

	// Output
	void PrintGraph(std::ostream& graph_output);
	datum AggregateLoss();

	// Status
	bool IsComplete() const;
private:
	void PrepareNode(NetGraphNode* node);
	void FeedForward(NetGraphNode* node);
	void BackPropagate(NetGraphNode* node);
	void InitializeNode(NetGraphNode* node);
  void InitializeWeights(NetGraphNode* node);
	std::vector<NetGraphNode*> nodes_;

	std::vector<NetGraphNode*> input_nodes_;
	std::vector<NetGraphNode*> output_nodes_;

	std::vector<NetGraphNode*> stat_nodes_;
	std::vector<NetGraphNode*> loss_nodes_;
	std::vector<NetGraphNode*> training_nodes_;

	int last_uid = -1;
};

}

#endif
