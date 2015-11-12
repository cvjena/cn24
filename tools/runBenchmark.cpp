/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file runBenchmark.cpp
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>

#include <cn24.h>

std::string hardcoded_net = "# Sample CNN for LabelMeFacade Dataset \
#anual rfx=34 rfy=34 factorx=4 factory=4 \
 \
# Network configuration \
?convolutional kernels=16 size=7x7 \
?maxpooling size=4x4 \
?tanh \
 \
?convolutional kernels=12 size=5x5 \
?maxpooling size=2x2 \
?tanh \
 \
?convolutional kernels=96 size=5x5 \
?tanh \
 \
?fullyconnected neurons=512 \
?tanh \
 \
?fullyconnected neurons=192 \
?tanh \
 \
?fullyconnected neurons=(o) \
?output \
 \
# Learning settings \
l1=0.000 \
l2=0.0008 \
lr=0.02 \
gamma=0.003 \
momentum=0.9 \
exponent=0.75 \
iterations=100 \
sbatchsize=24 \
pbatchsize=1 \
mu=1.75 \
eta=0.1 \
";

int main (int argc, char* argv[]) {
  // Initialize CN24
  Conv::System::Init(3);
  
  // Capture command line arguments
  std::string net_config_fname;
  if(argc > 1) {
    net_config_fname = std::string(argv[1]);
    LOGDEBUG << "Using user specified net: " << net_config_fname;
  }
  unsigned int CLASSES = 10;
  unsigned int INPUTMAPS = 3;
  

  std::istream* net_config_stream;
  
  if(argc > 1) {
    // Open network and dataset configuration files
    std::ifstream* net_config_file = new std::ifstream(net_config_fname,std::ios::in);
    
    if(!net_config_file->good()) {
      FATAL("Cannot open net configuration file!");
    }
      net_config_stream = net_config_file;
  } else {
    LOGINFO << "Using net: \n" << hardcoded_net << "\n";
    std::stringstream* ss = new std::stringstream(hardcoded_net);
    net_config_stream = ss;
  }
  
  // Parse network configuration file
  Conv::ConfigurableFactory* factory = new Conv::ConfigurableFactory(*net_config_stream, 238238, false);
  
  // Set image dimensions
  unsigned int width = 512;
  unsigned int height = 512;

  
  Conv::Tensor data_tensor(1, width, height, INPUTMAPS);
  data_tensor.Clear();

  // Assemble net
	Conv::NetGraph graph;
  Conv::InputLayer input_layer(data_tensor);

	Conv::NetGraphNode input_node(&input_layer);
  input_node.is_input = true;

	graph.AddNode(&input_node);
	bool complete = factory->AddLayers(graph, Conv::NetGraphConnection(&input_node), CLASSES);
	if (!complete)
    FATAL("Failed completeness check, inspect model!");

	graph.Initialize();

  graph.SetIsTesting(true);
  LOGINFO << "Running initial feedforward pass..." << std::flush;
  graph.FeedForward();
  
	Conv::Tensor* net_output_tensor = &graph.GetDefaultOutputNode()->output_buffers[0].combined_tensor->data;
  Conv::Tensor image_output_tensor(1, net_output_tensor->width(), net_output_tensor->height(), 3);
  
  //LOGINFO << "Colorizing..." << std::flush;
  //dataset->Colorize(*net_output_tensor, image_output_tensor);
  

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
