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
#include <iomanip>
#include <cstdlib>

// TEST SETUP
unsigned int RANDOM_RUNS = 50;
unsigned int SAMPLES = 2, WIDTH = 9, HEIGHT = 6, MAPS = 3;
Conv::datum epsilon = 0.005;

std::vector<std::pair<std::string, bool>> test_layers_and_runs = {
  {"{\"layer\":{\"type\":\"yolo_output\",\"yolo_configuration\":{\"boxes_per_cell\":1,\"horizontal_cells\":2,\"vertical_cells\":2}}}",true},
  {"{\"layer\":{\"type\":\"dropout\",\"dropout_fraction\":0.3}}",true},
  {"{\"layer\":{\"type\":\"simple_maxpooling\",\"size\":[3,3]}}",false},
  {"{\"layer\":{\"type\":\"simple_maxpooling\",\"size\":[3,2]}}",false},
  {"{\"layer\":{\"type\":\"advanced_maxpooling\",\"size\":[3,3]}}",false},
  {"{\"layer\":{\"type\":\"advanced_maxpooling\",\"size\":[3,2]}}",false},
  {"{\"layer\":{\"type\":\"advanced_maxpooling\",\"size\":[3,3],\"stride\":[2,2]}}",false},
  {"{\"layer\":{\"type\":\"convolution\",\"size\":[3,3],\"kernels\":3}}",true},
  {"{\"layer\":{\"type\":\"convolution\",\"size\":[3,3],\"stride\":[2,2],\"kernels\":3}}",true},
	{"{\"layer\":{\"type\":\"convolution\",\"size\":[3,3],\"pad\":[2,2],\"kernels\":3}}",true},
	{"{\"layer\":{\"type\":\"convolution\",\"size\":[3,3],\"stride\":[2,2],\"pad\":[2,2],\"kernels\":3}}",true},
	{"{\"layer\":{\"type\":\"convolution\",\"size\":[3,3],\"stride\":[2,2],\"pad\":[2,2],\"kernels\":9,\"group\":3}}",true},
  {"{\"layer\":{\"type\":\"convolution\",\"size\":[3,3],\"group\":3,\"kernels\":9}}",true},
  {"{\"layer\":{\"type\":\"hmax\",\"mu\":0.1,\"weight\":0.0}}",false},
  {"{\"layer\":{\"type\":\"hmax\",\"mu\":0.1,\"weight\":0.2}}",false},
  {"{\"layer\":{\"type\":\"sparsity_relu\",\"lambda\":0.1,\"kl_weight\":0.0,\"other_weight\":1.0,\"alpha\":3.0}}",true},
	{"{\"layer\":{\"type\":\"upscale\",\"size\":[3,3]}}",false},
  {"{\"layer\":\"tanh\"}",false},
  {"{\"layer\":\"sigm\"}",false},
  {"{\"layer\":\"relu\"}",false},
	{"{\"layer\":\"leaky\"}",false},
  {"{\"layer\":{\"type\":\"gradient_accumulation\",\"outputs\":2}}",false},
  {"{\"layer\":{\"type\":\"resize\",\"border\":[2,2]}}",false}
};

// UTILITIES
Conv::datum SimpleSumLoss(Conv::Layer* layer, const Conv::Tensor& tensor) {
	UNREFERENCED_PARAMETER(layer);
#ifdef BUILD_OPENCL
  ((Conv::Tensor&)tensor).MoveToCPU();
#endif
  Conv::datum sum = 0;
  
  for (unsigned int e = 0; e < tensor.elements(); e++) {
    const Conv::datum element = tensor.data_ptr_const()[e];
    sum += fabs(element);
  }
  
  return sum;
}

Conv::datum SimpleSumLoss(Conv::Layer* layer, const std::vector<Conv::CombinedTensor*>& outputs) {
  Conv::datum sum = 0;
  
  for (unsigned int o = 0; o < outputs.size(); o++) {
    sum += SimpleSumLoss(layer, outputs[o]->data);
  }
  
  Conv::LossFunctionLayer* loss_layer = dynamic_cast<Conv::LossFunctionLayer*>(layer);
  if (loss_layer != NULL)
    sum += loss_layer->CalculateLossFunction();
  
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

	Conv::ClassManager class_manager;
	class_manager.RegisterClassByName("test1", 0, 1);
	class_manager.RegisterClassByName("test2", 0, 1);
	class_manager.RegisterClassByName("test3", 0, 1);
	class_manager.RegisterClassByName("test4", 0, 1);

  Conv::NetStatus net_status;
  net_status.SetIsTesting(true);
	net_status.SetIsGradientTesting(true);
  
  bool test_failed = false;
  
  Conv::CombinedTensor input_data(SAMPLES, WIDTH, HEIGHT, MAPS);
	
	if(argc > 1) {
		std::string random_run_count_s = argv[1];
		unsigned int random_run_count = std::atoi(random_run_count_s.c_str());
		if(random_run_count > 0)
			RANDOM_RUNS = random_run_count;
	}
  
  /*std::vector<Conv::JSON> test_layers;
  
  if (argc > 1) {
    std::string test_layer = argv[1];
    test_layers.push_back(Conv::JSON::parse(test_layer));
  } else {
    for (std::pair<std::string, unsigned int>& layer_pair : test_layers_and_runs) {
      std::string& layer_descriptor = layer_pair.first;
      unsigned int runs = layer_pair.second;
      for(unsigned int i = 0; i < runs; i++) {
        // Inject random seeds
        Conv::JSON injected_descriptor = Conv::LayerFactory::InjectSeed(Conv::JSON::parse(layer_descriptor), seed_generator());
        test_layers.push_back(injected_descriptor);
      }
    }
  }*/
  
  for (std::pair<std::string, bool>& layer_pair : test_layers_and_runs) {
		Conv::JSON raw_layer_descriptor = Conv::JSON::parse(layer_pair.first);
		unsigned int failed_input_gradient_tests = 0;
		unsigned int failed_weight_gradient_tests = 0;
		unsigned int total_runs = layer_pair.second ? RANDOM_RUNS : 1;
    unsigned int total_input_tests = 0;
		unsigned int total_weight_tests = 0;

		LOGINFO << "Testing layer: " << raw_layer_descriptor.dump() << " with " << total_runs << " runs..." << std::flush;
			
		for(unsigned int current_run = 0; current_run < total_runs; current_run++) {
			Conv::JSON layer_descriptor = Conv::LayerFactory::InjectSeed(raw_layer_descriptor, seed_generator());
			bool data_sign = seed_generator() % 2 == 0;
#ifdef BUILD_OPENCL
      input_data.data.MoveToCPU();
#endif
			for(unsigned int e = 0; e < input_data.data.elements(); e++) {
				if(data_sign)
					input_data.data.data_ptr()[e] = dist(seed_generator);
				else
					input_data.data.data_ptr()[e] = -dist(seed_generator);
			}
			input_data.delta.Clear(0.0);

			Conv::Layer* layer;
			if(Conv::LayerFactory::ExtractLayerType(layer_descriptor).compare("yolo_output") == 0) {
				layer = new Conv::YOLODynamicOutputLayer(Conv::LayerFactory::ExtractConfiguration(layer_descriptor), &class_manager);
				input_data.data.Reshape(SAMPLES, 1, 1, WIDTH * HEIGHT * MAPS);
			} else {
				layer = Conv::LayerFactory::ConstructLayer(layer_descriptor);
				input_data.data.Reshape(SAMPLES, WIDTH, HEIGHT, MAPS);
			}
			if(layer == nullptr) {
				test_failed = true;
				LOGINFO << "    Constructing...";
				LOGERROR << "        FAILED";
				current_run = total_runs;
				continue;
			}

#ifdef BUILD_OPENCL
      if(layer->IsGPUMemoryAware()) {
				if(seed_generator() % 4 < 2)
          input_data.data.MoveToGPU();
        LOGDEBUG << "    OpenCL aware!";
			}
#endif
			LOGDEBUG << "    Description: " << layer->GetLayerDescription();
			LOGDEBUG << "    Configuration: " << layer->GetLayerConfiguration().dump();
			
			std::vector<Conv::CombinedTensor*> outputs;
			bool createoutputs_success = layer->CreateOutputs({&input_data}, outputs);
			if(!createoutputs_success) {
				test_failed = true;
				LOGINFO << "    Creating outputs...";
				LOGERROR << "       FAILED";
				current_run = total_runs;
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
				current_run = total_runs;
				continue;
			}
			
			layer->OnLayerConnect({}, false);
			
			if(layer->parameters().size() == 0) {
				LOGDEBUG << "    Layer has no weights";
			}
			
			// Function pointers for external gradient check
			void (*WriteLossDeltas)(const std::vector<Conv::CombinedTensor*>&) =
				SimpleSumLossGradient;
			Conv::datum (*CalculateLoss)(Conv::Layer*, const std::vector<Conv::CombinedTensor*>&) =
				SimpleSumLoss;
			
			for(Conv::CombinedTensor* weights : layer->parameters()) {
        LOGDEBUG << "    Gradient test (weight set " << weights->data << ")..." << std::flush;
				total_weight_tests++;
				bool gradient_success = Conv::GradientTester::DoGradientTest(layer, weights->data, weights->delta, outputs, epsilon, WriteLossDeltas, CalculateLoss);
				if(!gradient_success) {
					failed_weight_gradient_tests++;
					continue;
				}
			}

			LOGDEBUG << "    Gradient test (inputs)..." << std::flush;
			total_input_tests++;
			bool gradient_success = Conv::GradientTester::DoGradientTest(layer, input_data.data, input_data.delta, outputs, epsilon, WriteLossDeltas, CalculateLoss);
			if(!gradient_success) {
				failed_input_gradient_tests++;
			}
			
			delete layer;
			for(Conv::CombinedTensor* output : outputs)
				delete output;
		}
		
		Conv::datum input_gradient_failure_rate = ((Conv::datum)failed_input_gradient_tests)/((Conv::datum)total_input_tests);
		Conv::datum weight_gradient_failure_rate = ((Conv::datum)failed_weight_gradient_tests)/((Conv::datum)total_weight_tests);
		
		if(input_gradient_failure_rate > 0.2) {
			test_failed = true;
			LOGINFO << "    Gradient test (inputs)...";
			LOGERROR << "        " << std::setprecision(3) << input_gradient_failure_rate * 100.0 << "% of runs failed the gradient test";
			LOGERROR << "        FAILED";
		}
		
		if(weight_gradient_failure_rate > 0.2) {
			test_failed = true;
			LOGINFO << "    Gradient test (weights)...";
			LOGERROR << "        " << std::setprecision(3) << weight_gradient_failure_rate * 100.0 << "% of runs failed the gradient test";
			LOGERROR << "        FAILED";
		}
		
		if(!test_failed) {
			std::cout << "OK (W:" << std::setprecision(3) << (1.0 - weight_gradient_failure_rate) * 100.0 << "%";
			std::cout << " I:" << std::setprecision(3) << (1.0 - input_gradient_failure_rate) * 100.0 << "%)";
		}
		
  }
  
  Conv::System::Shutdown();
  
  return test_failed ? -1 : 0;
}
