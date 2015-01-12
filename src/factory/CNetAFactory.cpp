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
#include "UpscaleLayer.h"
#include "ResizeLayer.h"
#include "SpatialPriorLayer.h"
#include "Factory.h"

namespace Conv {
int CNetAFactory::AddLayers ( Net& net, Connection data_layer_connection ) {
  std::mt19937 rand ( seed_ );

  // Needed for helper data
  int data_layer = data_layer_connection.net;
  Connection helper_connection = Connection ( data_layer, 2 );

  // FIXME This style leaks a lot of memory

  // Instantiate layers
  auto* layer0 = new ResizeLayer(22,22);
  auto* layer1 = new Conv::ConvolutionLayer ( 7, 7, 12, rand() );
  layer1->SetLocalLearningRate(6.0);
  layer1->SetBackpropagationEnabled ( false );
  auto* layer2 = new MaxPoolingLayer ( 2, 2 );
  auto* layer3 = new ReLULayer();
  auto* layer4 = new Conv::ConvolutionLayer ( 5, 5, 6, rand() );
  layer4->SetLocalLearningRate(1.6);
  auto* layer5 = new ReLULayer();
  auto* layer6 = new Conv::ConvolutionLayer ( 5, 5, 48, rand() );
  auto* layer7 = new ReLULayer();
  auto* layer7b = new SpatialPriorLayer();
  auto* layer8 = new Conv::ConvolutionLayer ( 1, 1, 192, rand() );
  auto* layer9 = new ReLULayer();
  auto* layer10 = new Conv::ConvolutionLayer ( 1, 1, 1, rand() );
  auto* layer11 = new TanhLayer();
  auto* layer12 = new UpscaleLayer( 2, 2);
  //auto* layer9 = new Conv::ConvolutionLayer ( 1, 1, 1, rand() );
  //auto* layer10 = new TanhLayer();

  // Add layers
  int layer0_id = net.AddLayer ( layer0, { data_layer_connection } );
  int layer1_id = net.AddLayer ( layer1, layer0_id );
  int layer2_id = net.AddLayer ( layer2, layer1_id );
  int layer3_id = net.AddLayer ( layer3, layer2_id );
  int layer4_id = net.AddLayer ( layer4, layer3_id );
  int layer5_id = net.AddLayer ( layer5, layer4_id );
  int layer6_id = net.AddLayer ( layer6, layer5_id );
  int layer7_id = net.AddLayer ( layer7, layer6_id );
  int layer7b_id = net.AddLayer ( layer7b, layer7_id );
  int layer8_id = net.AddLayer ( layer8, layer7b_id );
  int layer9_id = net.AddLayer ( layer9, layer8_id );
  int layer10_id = net.AddLayer ( layer10, layer9_id );
  int layer11_id = net.AddLayer ( layer11, layer10_id );
  int layer12_id = net.AddLayer ( layer12, layer11_id );

  // Configure first layer to disable backpropagation

  return layer12_id;
}

void CNetAFactory::InitOptimalSettings() {
  optimal_settings_.learning_rate = 1.0;
  optimal_settings_.momentum = 0.85;
  optimal_settings_.gamma = 0.07;
  optimal_settings_.exponent = 0.75;

  // DL_TANH* runs are with l1=;0
  // DL_RELU and DL_ALL are with l1=0.001
  optimal_settings_.l1_weight = 0.0001;
  optimal_settings_.l2_weight = 0.00005;
  optimal_settings_.iterations = 50;
  optimal_settings_.testing_ratio = 1;
}

}
