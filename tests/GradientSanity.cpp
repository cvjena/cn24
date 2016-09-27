/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  

/**
 * @file GradientSanity.cpp
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <chrono>
#include <random>

#include <cn24.h>

std::string hardcoded_net = "# Network configuration \n\
?convolutional kernels=8 size=7x7 \n\
?maxpooling size=2x2 \n\
 \n\
?convolutional kernels=12 size=5x5 \n\
?tanh \n\
 \n\
?fullyconnected neurons=64 \n\
?tanh \n\
 \n\
?fullyconnected neurons=(o) \n\
?output \n\
 \n\
# Learning settings \n\
l1=0.000 \n\
l2=0.0008 \n\
lr=0.02 \n\
gamma=0.003 \n\
momentum=0.9 \n\
exponent=0.75 \n\
iterations=100 \n\
sbatchsize=24 \n\
pbatchsize=2 \n\
mu=1.75 \n\
eta=0.1 \n\
";

int main (int argc, char* argv[]) {
  UNREFERENCED_PARAMETER(argc);
  UNREFERENCED_PARAMETER(argv);
  // Initialize CN24
  Conv::System::Init();
  
  // Set benchmark arguments
  unsigned int CLASSES = 10;
  unsigned int INPUTMAPS = 3;
  unsigned int width = 64;
  unsigned int height = 64;

  std::istream* net_config_stream;
  std::stringstream* ss = new std::stringstream(hardcoded_net);
  net_config_stream = ss;
  
  // Parse network configuration file
  Conv::ConfigurableFactory* factory = new Conv::ConfigurableFactory(*net_config_stream, 238238, false);

  Conv::Tensor data_tensor(factory->optimal_settings().pbatchsize, width, height, INPUTMAPS);
  data_tensor.Clear();

  // Generate random contents
  std::mt19937 rand(1337);
  std::uniform_real_distribution<Conv::datum> dist (0.0, 1.0);
  for(unsigned int e = 0; e < data_tensor.elements(); e++) {
    (data_tensor.data_ptr())[e] = dist(rand);
  }

  // Assemble net
	Conv::NetGraph graph;
  Conv::InputLayer input_layer(data_tensor);

	Conv::NetGraphNode input_node(&input_layer);
  input_node.is_input = true;

	graph.AddNode(&input_node);
	bool complete = factory->AddLayers(graph, Conv::NetGraphConnection(&input_node), CLASSES);
	if (!complete)
    FATAL("Failed completeness check, inspect model!");
	factory->InitOptimalSettings();

	LOGINFO << "Initializing net, this may take a while..." << std::flush;
	graph.Initialize();
  graph.SetIsTesting(true);
  
  graph.FeedForward();
	graph.BackPropagate();
	
  // Gradient check
  Conv::GradientTester::TestGradient(graph, 999, true);
  
  LOGEND;
  return 0;
}
