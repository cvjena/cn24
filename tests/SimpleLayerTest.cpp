/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>

#include <vector>
#include <utility>
#include <string>
#include <cmath>
#include <random>

// TEST SETUP
unsigned int RANDOM_RUNS = 15;
unsigned int SAMPLES = 2, WIDTH = 9, HEIGHT = 6, MAPS = 3;
Conv::datum epsilon = 0.005;

std::vector<std::pair<std::string, unsigned int>> test_layers_and_runs = {
  {"maxpooling(size=3x3)",1},
  {"maxpooling(size=3x2)",1},
  {"amaxpooling(size=3x3)",1},
  {"amaxpooling(size=3x2)",1},
  {"amaxpooling(size=3x3 stride=2x2)",1},
  {"convolution(size=3x3 kernels=3)",RANDOM_RUNS},
  {"convolution(size=3x3 stride=2x2 kernels=3)",RANDOM_RUNS},
  {"convolution(size=3x3 group=3 kernels=9)",RANDOM_RUNS},
  {"tanh",1},{"sigm",1},{"relu",1},
  {"gradientaccumulation(outputs=2)",1},
  {"resize(border=2x2)",1}
};

// UTILITIES
Conv::datum SimpleSumLoss(const Conv::Tensor& tensor) {
#ifdef BUILD_OPENCL
  tensor.MoveToCPU();
#endif
  Conv::datum sum = 0;
  
  for (unsigned int e = 0; e < tensor.elements(); e++) {
    const Conv::datum element = tensor.data_ptr_const()[e];
    sum += fabs(element);
  }
  
  return sum;
}

Conv::datum SimpleSumLoss(const std::vector<Conv::CombinedTensor*>& outputs) {
  Conv::datum sum = 0;
  
  for (unsigned int o = 0; o < outputs.size(); o++) {
    sum += SimpleSumLoss(outputs[o]->data);
  }
  
  return sum;
}

void SimpleSumLossGradient(const std::vector<Conv::CombinedTensor*>& outputs) {
  for (unsigned int o = 0; o < outputs.size(); o++) {
    Conv::Tensor& tensor = outputs[o]->data;
    Conv::Tensor& delta_tensor = outputs[o]->delta;
#ifdef BUILD_OPENCL
    tensor.MoveToCPU();
    delta_tensor.MoveToCPU();
#endif
    for (unsigned int e = 0; e < tensor.elements(); e++) {
      const Conv::datum element = tensor.data_ptr_const()[e];
      const Conv::datum gradient = element > 0.0 ? 1.0 : -1.0;
      delta_tensor.data_ptr()[e] = gradient;
    }
  }
}

int main(int argc, char* argv[]) {
  if((argc > 1 && std::string("-v").compare(argv[1]) == 0) || argc > 2) {
    Conv::System::Init(3);
  } else {
    Conv::System::Init();
  }
  
  std::mt19937 seed_generator(93023);
  std::uniform_real_distribution<Conv::datum> dist(1.0, 2.0);
  
  Conv::NetStatus net_status;
  net_status.SetIsTesting(true);
  
  bool test_failed = false;
  
  Conv::CombinedTensor input_data(SAMPLES, WIDTH, HEIGHT, MAPS);
  
  std::vector<std::string> test_layers;
  
  if (argc > 1) {
    std::string test_layer = argv[1];
    test_layers.push_back(test_layer);
  } else {
    for (std::pair<std::string, unsigned int>& layer_pair : test_layers_and_runs) {
      std::string& layer_descriptor = layer_pair.first;
      unsigned int runs = layer_pair.second;
      for(unsigned int i = 0; i < runs; i++) {
        // Inject random seeds
        std::string injected_descriptor = Conv::LayerFactory::InjectSeed(layer_descriptor, seed_generator());
        test_layers.push_back(injected_descriptor);
      }
    }
  }
  
  for (std::string& layer_descriptor : test_layers) {
    bool data_sign = seed_generator() % 2 == 0;
    for(unsigned int e = 0; e < input_data.data.elements(); e++) {
      if(data_sign)
        input_data.data.data_ptr()[e] = dist(seed_generator);
      else
        input_data.data.data_ptr()[e] = -dist(seed_generator);
    }
    input_data.delta.Clear(0.0);
    
    LOGINFO << "Testing layer: " << layer_descriptor;
    Conv::Layer* layer = Conv::LayerFactory::ConstructLayer(layer_descriptor);
    if(layer == nullptr) {
      test_failed = true;
      LOGERROR << "        FAILED";
      continue;
    }
    
    LOGDEBUG << "    Description: " << layer->GetLayerDescription();
    LOGDEBUG << "    Configuration: " << layer->GetLayerConfiguration();
    
    if (Conv::LayerFactory::ExtractConfiguration(layer_descriptor).length() > 0
        && layer->GetLayerConfiguration().length() == 0) {
      test_failed = true;
      LOGINFO << "    Testing configuration loss...";
      LOGERROR << "       FAILED";
      continue;
    }
    
    std::vector<Conv::CombinedTensor*> outputs;
    bool createoutputs_success = layer->CreateOutputs({&input_data}, outputs);
    if(!createoutputs_success) {
      test_failed = true;
      LOGINFO << "    Creating outputs...";
      LOGERROR << "       FAILED";
      continue;
    }
    
    for(Conv::CombinedTensor* output : outputs) {
      LOGDEBUG << "        Output: " << output->data;
    }
    
    bool connect_success = layer->Connect({&input_data}, outputs, &net_status);
    if(!connect_success) {
      test_failed = true;
      LOGINFO << "    Connecting...";
      LOGERROR << "        FAILED";
      continue;
    }
    
    layer->OnLayerConnect({});
    
    if(layer->parameters().size() == 0) {
      LOGDEBUG << "    Layer has no weights";
    }
    
    // Function pointers for external gradient check
    void (*WriteLossDeltas)(const std::vector<Conv::CombinedTensor*>&) =
      SimpleSumLossGradient;
    Conv::datum (*CalculateLoss)(const std::vector<Conv::CombinedTensor*>&) =
      SimpleSumLoss;
    
    for(Conv::CombinedTensor* weights : layer->parameters()) {
      bool gradient_success = Conv::GradientTester::DoGradientTest(layer, weights->data, weights->delta, outputs, epsilon, WriteLossDeltas, CalculateLoss);
      if(!gradient_success) {
        test_failed = true;
        LOGINFO << "    Gradient test (weights)...";
        LOGERROR << "        FAILED";
        continue;
      }
    }
    
    bool gradient_success = Conv::GradientTester::DoGradientTest(layer, input_data.data, input_data.delta, outputs, epsilon, WriteLossDeltas, CalculateLoss);
    if(!gradient_success) {
      test_failed = true;
      LOGINFO << "    Gradient test (inputs)...";
      LOGERROR << "        FAILED";
      continue;
    }
    
    delete layer;
    for(Conv::CombinedTensor* output : outputs)
      delete output;
  }
  
  Conv::System::Shutdown();
  
  return test_failed ? -1 : 0;
}
