/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <cn24.h>

#include <vector>
#include <string>
#include <cmath>
#include <random>

// TEST SETUP
std::vector<std::string> test_layers_noseed = {
  "maxpooling(size=3x3)",
  "amaxpooling(size=3x3)",
  "amaxpooling(size=3x3 stride=2x2)",
  "convolution(size=3x3 kernels=3)",
  "convolution(size=3x3 stride=2x2 kernels=3)",
  "convolution(size=3x3 group=3 kernels=9)",
  "tanh","sigm","relu"
};

unsigned int SAMPLES = 2, WIDTH = 9, HEIGHT = 6, MAPS = 3;
unsigned int SEEDS = 3;
Conv::datum epsilon = 0.005;

// UTILITIES
Conv::datum SimpleSumLoss(const Conv::Tensor& tensor) {
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
    for (unsigned int e = 0; e < tensor.elements(); e++) {
      const Conv::datum element = tensor.data_ptr_const()[e];
      const Conv::datum gradient = element > 0.0 ? 1.0 : -1.0;
      delta_tensor.data_ptr()[e] = gradient;
    }
  }
}

namespace Conv {
  bool DoGradientTest(Conv::Layer* layer, Conv::Tensor& data, Conv::Tensor& delta, std::vector<Conv::CombinedTensor*>& outputs, Conv::datum epsilon) {
    layer->FeedForward();
    SimpleSumLossGradient(outputs);
    layer->BackPropagate();
    
    unsigned int elements = data.elements();
    unsigned int okay = 0;

    // Weight gradient test
    for (unsigned int w = 0; w < data.elements(); w++) {
      const Conv::datum weight = data.data_ptr_const()[w];
      const Conv::datum gradient = delta.data_ptr_const()[w];

      // Using central diff
      data.data_ptr()[w] = weight + epsilon;
      layer->FeedForward();
      const Conv::datum forward_loss = SimpleSumLoss(outputs);

      data.data_ptr()[w] = weight - epsilon;
      layer->FeedForward();
      const Conv::datum backward_loss = SimpleSumLoss(outputs);

      const Conv::datum fd_gradient = (forward_loss - backward_loss) / (2.0 * epsilon);
      data.data_ptr()[w] = weight;

      const Conv::datum ratio = fd_gradient / gradient;
      if(ratio > 1.2 || ratio < 0.8) {
        LOGDEBUG << "BP Grad : " << gradient;
        LOGDEBUG << "FD Grad : " << fd_gradient;
        LOGDEBUG << "Ratio   : " << ratio;
        LOGDEBUG << "Diff    : " << gradient - fd_gradient;
      } else {
        okay++;
      }
    }
    if(okay != elements) {
      double success_rate = (double)okay/(double)elements;
      if(success_rate > 0.95)
        return true;
      else {
        LOGERROR << okay << " of " << elements << " gradients okay - " << std::setprecision(3) << 100.0 * (double)okay/(double)elements << "%";
        return false;
      }
    } else {
      return true;
    }
  }
}

int main(int argc, char* argv[]) {
  if(argc > 1)
    Conv::System::Init(3);
  else
    Conv::System::Init();
  
  std::mt19937 seed_generator(93023);
  std::uniform_real_distribution<Conv::datum> dist(1.0, 2.0);
  
  Conv::NetStatus net_status;
  net_status.SetIsTesting(true);
  
  bool test_failed = false;
  
  Conv::CombinedTensor input_data(SAMPLES, WIDTH, HEIGHT, MAPS);
  
  std::vector<std::string> test_layers;
  
  // Inject random seeds
  for (std::string& layer_descriptor : test_layers_noseed) {
    for(unsigned int i = 0; i < SEEDS; i++) {
      std::string injected_descriptor = Conv::LayerFactory::InjectSeed(layer_descriptor, seed_generator());
      test_layers.push_back(injected_descriptor);
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
    
    for(Conv::CombinedTensor* weights : layer->parameters()) {
      bool gradient_success = Conv::DoGradientTest(layer, weights->data, weights->delta, outputs, epsilon);
      if(!gradient_success) {
        test_failed = true;
        LOGINFO << "    Gradient test (weights)...";
        LOGERROR << "        FAILED";
        continue;
      }
    }
    
    bool gradient_success = Conv::DoGradientTest(layer, input_data.data, input_data.delta, outputs, epsilon);
    if(!gradient_success) {
      test_failed = true;
      LOGINFO << "    Gradient test (inputs)...";
      LOGERROR << "        FAILED";
      continue;
    }
  }
  
  LOGEND;
  return test_failed ? -1 : 0;
}