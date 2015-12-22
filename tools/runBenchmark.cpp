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
#include <chrono>
#include <random>

#include <cn24.h>

std::string hardcoded_net = "# Sample CNN for LabelMeFacade Dataset \n\
#anual rfx=34 rfy=34 factorx=4 factory=4 \n\
 \n\
# Network configuration \n\
?convolutional kernels=16 size=7x7 \n\
?maxpooling size=4x4 \n\
?tanh \n\
 \n\
?convolutional kernels=12 size=5x5 \n\
?maxpooling size=2x2 \n\
?tanh \n\
 \n\
?convolutional kernels=96 size=5x5 \n\
?tanh \n\
 \n\
?fullyconnected neurons=512 \n\
?tanh \n\
 \n\
?fullyconnected neurons=192 \n\
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

std::string hardcoded_net_ci = "# Sample CNN for LabelMeFacade Dataset \n\
#anual rfx=34 rfy=34 factorx=4 factory=4 \n\
 \n\
# Network configuration \n\
?convolutional kernels=8 size=7x7 \n\
?maxpooling size=2x2 \n\
 \n\
?convolutional kernels=4 size=5x5 \n\
?tanh \n\
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
  // Initialize stat descriptor
  Conv::StatDescriptor fps;
  fps.nullable = false;
  fps.description = "Throughput";
  fps.unit = "frames/s";
  fps.init_function = [] (Conv::Stat& stat) {
    stat.is_null = false;
    stat.value = 0.0;
  };
  fps.output_function = [] (Conv::HardcodedStats& hc_stats, Conv::Stat& stat) {
    Conv::Stat return_stat = stat;
    return_stat.value = stat.value / hc_stats.seconds_elapsed;
    return return_stat;
  };
  fps.update_function = [] (Conv::Stat& stat, double user_value) {
    stat.value += user_value;
  };
  
  // Initialize CN24
  Conv::System::Init();
  
  // Register ConsoleStatSink
  Conv::ConsoleStatSink sink;
  Conv::System::stat_aggregator->RegisterSink(&sink);
  
  // Register fps counter
  Conv::System::stat_aggregator->RegisterStat(&fps);
  
  // Capture command line arguments
  std::string net_config_fname;
  if(argc > 1) {
    net_config_fname = std::string(argv[1]);
    LOGDEBUG << "Using user specified net: " << net_config_fname;
  }
  
  // Set benchmark arguments
  unsigned int CLASSES = 10;
  unsigned int INPUTMAPS = 3;
	unsigned int BENCHMARK_PASSES_FWD = 30;
	unsigned int BENCHMARK_PASSES_BWD = 15;
  unsigned int width = 512;
  unsigned int height = 512;

  std::istream* net_config_stream;
  
  if(argc > 1 && net_config_fname.compare(0,4,"--ci") != 0) {
    // Open network and dataset configuration files
    std::ifstream* net_config_file = new std::ifstream(net_config_fname,std::ios::in);
    
    if(!net_config_file->good()) {
      FATAL("Cannot open net configuration file!");
    }
      net_config_stream = net_config_file;
  } else {
    if(net_config_fname.compare(0,4,"--ci") == 0) {
      LOGINFO << "Using hardcoded net for continuous integration.";
      std::stringstream* ss = new std::stringstream(hardcoded_net_ci);
      net_config_stream = ss;
      width = 64;
      height = 64;
    } else {
      LOGINFO << "Using hardcoded net.";
      std::stringstream* ss = new std::stringstream(hardcoded_net);
      net_config_stream = ss;
    }
  }
  
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
  
  // Initialize StatAggregator
  Conv::System::stat_aggregator->Initialize();
  
  graph.FeedForward();
	graph.BackPropagate();
	
	LOGINFO << "Benchmark information";
	LOGINFO << "=====================";
	
	LOGINFO << "Input width    : " << width;
	LOGINFO << "Input height   : " << height;
	LOGINFO << "Parallel inputs: " << factory->optimal_settings().pbatchsize;
	LOGINFO << "=====================";
	
	LOGINFO << "Running forward benchmark...\n" << std::flush;
  Conv::System::stat_aggregator->StartRecording();
	{
		for(unsigned int p = 0; p < BENCHMARK_PASSES_FWD; p++) {
			graph.FeedForward();
      Conv::System::stat_aggregator->Update(fps.stat_id, (double)(factory->optimal_settings().pbatchsize));
			std::cout << "." << std::flush;
		}
		std::cout << "\n";
	}
  Conv::System::stat_aggregator->StopRecording();
  Conv::System::stat_aggregator->Generate();
	Conv::System::stat_aggregator->Reset();
  
  graph.SetIsTesting(false);
	LOGINFO << "Running forward+backward benchmark...\n" << std::flush;
  Conv::System::stat_aggregator->StartRecording();
	{
		for(unsigned int p = 0; p < BENCHMARK_PASSES_BWD; p++) {
			graph.FeedForward();
			graph.BackPropagate();
      Conv::System::stat_aggregator->Update(fps.stat_id, (double)(factory->optimal_settings().pbatchsize));
			std::cout << "." << std::flush;
		}
		std::cout << "\n";
	}
  Conv::System::stat_aggregator->StopRecording();
  Conv::System::stat_aggregator->Generate();
  Conv::System::stat_aggregator->Reset();
  
  // Gradient check
	LOGINFO << "Running sparse gradient check..." << std::flush;
  auto start_time = std::chrono::system_clock::now();
  Conv::GradientTester::TestGradient(graph, 999, true);
  
  auto stop_time = std::chrono::system_clock::now();
  std::chrono::duration<double> t_diff = stop_time - start_time;
  LOGRESULT << "Gradient check took " << t_diff.count() << " seconds." << LOGRESULTEND;
  
  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
