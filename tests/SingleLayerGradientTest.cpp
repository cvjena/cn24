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

std::vector<std::string> test_layers = {
  "convolution(size=3x3 kernels=3 seed=5)",
  "convolution(size=3x3 stride=2x2 kernels=3 seed=5)",
  "convolution(size=3x3 group=3 kernels=9 seed=5)",
  "tanh","sigm","relu"
};

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
        LOGDEBUG << "Expected: " << gradient << ", FD Grad : " << fd_gradient;
        LOGDEBUG << "Ratio: " << ratio;
      } else {
        okay++;
      }
    }
    if(okay != elements) {
      LOGDEBUG << okay << " of " << elements << " gradients okay - " << std::setprecision(3) << 100.0 * (double)okay/(double)elements << "%";
      double success_rate = (double)okay/(double)elements;
      if(success_rate > 0.95)
        return true;
      else
        return false;
    } else {
      return true;
    }
  }
}

int main(int argc, char* argv[]) {
  Conv::System::Init();
  
  Conv::NetStatus net_status;
  net_status.SetIsTesting(true);
  
  unsigned int SAMPLES = 2, WIDTH = 7, HEIGHT = 7, MAPS = 3;
  Conv::datum epsilon = 0.005;
  
  bool test_failed = false;
  
  Conv::CombinedTensor input_data(SAMPLES, WIDTH, HEIGHT, MAPS);
  
  for (std::string& layer_descriptor : test_layers) {
    input_data.data.Clear(2.0);
    input_data.delta.Clear(0.0);
    
    LOGINFO << "Testing layer: " << layer_descriptor;
    Conv::Layer* layer = Conv::LayerFactory::ConstructLayer(layer_descriptor);
    if(layer == nullptr) {
      test_failed = true;
      LOGERROR << "        FAILED";
      continue;
    }
    
    LOGINFO << "    Description: " << layer->GetLayerDescription();
    
    LOGINFO << "    Creating outputs...";
    std::vector<Conv::CombinedTensor*> outputs;
    bool createoutputs_success = layer->CreateOutputs({&input_data}, outputs);
    if(!createoutputs_success) {
      test_failed = true;
      LOGERROR << "       FAILED";
      continue;
    }
    
    for(Conv::CombinedTensor* output : outputs) {
      LOGINFO << "        " << output->data;
    }
    
    LOGINFO << "    Connecting...";
    bool connect_success = layer->Connect({&input_data}, outputs, &net_status);
    if(!connect_success) {
      test_failed = true;
      LOGERROR << "        FAILED";
      continue;
    }
    
    layer->OnLayerConnect({});
    
    LOGINFO << "    Gradient test (weights)...";
    for(Conv::CombinedTensor* weights : layer->parameters()) {
      bool gradient_success = Conv::DoGradientTest(layer, weights->data, weights->delta, outputs, epsilon);
      if(!gradient_success) {
        test_failed = true;
        LOGERROR << "        FAILED";
        continue;
      }
    }
    
    LOGINFO << "    Gradient test (inputs)...";
    bool gradient_success = Conv::DoGradientTest(layer, input_data.data, input_data.delta, outputs, epsilon);
    if(!gradient_success) {
      test_failed = true;
      LOGERROR << "        FAILED";
      continue;
    }
  }
  
  LOGEND;
  return test_failed ? -1 : 0;
}