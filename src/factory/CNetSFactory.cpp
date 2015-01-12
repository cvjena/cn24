/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
#include <random>
#include "Net.h"
#include "ConvolutionLayer.h"
#include "MaxPoolingLayer.h"
#include "ConcatLayer.h"
#include "NonLinearityLayer.h"
#include "FlattenLayer.h"
#include "FullyConnectedLayer.h"
#include "Factory.h"

namespace Conv {
  int CNetSFactory::AddLayers (Net& net, Connection data_layer_connection,
    const unsigned int output_classes ) {
    std::mt19937 rand (seed_);
    
    // Needed for helper data
    int data_layer = data_layer_connection.net;
    Connection helper_connection = Connection (data_layer, 2);
    
    // FIXME This style leaks a lot of memory
    
    // Instantiate layers
    auto* layer1  = new Conv::ConvolutionLayer (7, 7, 16, rand());
    auto* layer2 = new MaxPoolingLayer (2, 2);
    auto* layer3 = new TanhLayer();
    auto* layer4 = new Conv::ConvolutionLayer (7, 7, 16, rand());
    auto* layer5 = new TanhLayer();
    auto* layer6 = new Conv::ConvolutionLayer (7, 7, 16, rand());
    auto* layer7 = new TanhLayer();
    
    // Fully connected layers
    auto* layer8 = new FlattenLayer();
    auto* layer9 = new FullyConnectedLayer (192, rand());
    auto* layer10 = new TanhLayer();
    auto* layer11 = new ConcatLayer();
    auto* layer12 = new FullyConnectedLayer(192, rand());
    auto* layer13 = new TanhLayer();
    auto* layer14 = new FullyConnectedLayer (output_classes, rand());
    auto* layer15 = new SigmoidLayer();
    
    // Add layers
    int layer1_id = net.AddLayer (layer1, { data_layer_connection });
    int layer2_id = net.AddLayer (layer2, layer1_id);
    int layer3_id = net.AddLayer (layer3, layer2_id);
    int layer4_id = net.AddLayer (layer4, layer3_id);
    int layer5_id = net.AddLayer (layer5, layer4_id);
    int layer6_id = net.AddLayer (layer6, layer5_id);
    int layer7_id = net.AddLayer (layer7, layer6_id);
    int layer8_id = net.AddLayer (layer8, layer7_id);
    int layer9_id = net.AddLayer (layer9, layer8_id);
    int layer10_id = net.AddLayer (layer10, layer9_id);
    int layer11_id = net.AddLayer(layer11, { helper_connection, Connection(layer10_id)});
    int layer12_id = net.AddLayer (layer12, layer11_id);
    int layer13_id = net.AddLayer (layer13, layer12_id);
    int layer14_id = net.AddLayer (layer14, layer13_id);
    int layer15_id = net.AddLayer (layer15, layer14_id);
    
    // Configure first layer to disable backpropagation
    layer1->SetBackpropagationEnabled (false);
    
    return layer15_id;
  }
  
  void CNetSFactory::InitOptimalSettings() {
  optimal_settings_.learning_rate = 0.02;
  optimal_settings_.momentum = 0.9;
  optimal_settings_.gamma = 0.000030;
  optimal_settings_.exponent = 0.75;
  
  // DL_TANH* runs are with l1=;0
  // DL_RELU and DL_ALL are with l1=0.001
  optimal_settings_.l1_weight = 0.00;
  optimal_settings_.l2_weight = 0.0008;
  optimal_settings_.iterations = 10000;
  optimal_settings_.testing_ratio = 1;
  }
  
}
