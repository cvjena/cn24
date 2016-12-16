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
unsigned int RANDOM_RUNS = 1;
unsigned int SAMPLES = 2, WIDTH = 1, HEIGHT = 1, MAPS = 3 * 3 * (5 * 2 + 4);
Conv::datum epsilon = 0.005;

static Conv::YOLOLossLayer* global_loss_layer = nullptr;

void WriteLossDeltas(const std::vector<Conv::CombinedTensor*>& v) {

}

Conv::datum CalculateLoss(Conv::Layer* layer, const std::vector<Conv::CombinedTensor*>& v) {
  return global_loss_layer->CalculateLossFunction();
}

int main(int argc, char** argv) {
  if((argc > 1 && std::string("-v").compare(argv[1]) == 0) || argc > 2) {
    Conv::System::Init(3);
  } else {
    Conv::System::Init();
  }

  std::mt19937 seed_generator(93023);
  std::uniform_real_distribution<Conv::datum> dist(-0.1, 1.1);

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
  Conv::CombinedTensor label_data(SAMPLES);
  Conv::CombinedTensor weight_data(SAMPLES);
  weight_data.data[0] = 1;
  weight_data.data[1] = 1;

  Conv::JSON yolo_configuration = Conv::JSON::parse("{\"boxes_per_cell\":2,\"horizontal_cells\":3,\"vertical_cells\":3}");
  Conv::YOLOLossLayer loss_layer(yolo_configuration);
  global_loss_layer = &loss_layer;

  Conv::BoundingBox bbox1(0.5, 0.2, 0.2, 0.7); bbox1.c = 2;
  Conv::BoundingBox bbox2(0.9, 0.1, 0.5, 0.1); bbox1.c = 3;
  std::vector<Conv::BoundingBox> vec1 = {bbox1};
  std::vector<Conv::BoundingBox> vec2 = {bbox2};
  Conv::DatasetMetadataPointer* metadata = new Conv::DatasetMetadataPointer[SAMPLES];
  metadata[0] = (Conv::DatasetMetadataPointer)&vec1;
  metadata[1] = (Conv::DatasetMetadataPointer)&vec2;
  label_data.metadata = metadata;


  for(unsigned int current_run = 0; current_run < RANDOM_RUNS; current_run++) {
    for(unsigned int e = 0; e < input_data.data.elements(); e++) {
      input_data.data.data_ptr()[e] = dist(seed_generator);
    }
    input_data.delta.Clear(0.0);
    std::vector<Conv::CombinedTensor*> tmp;
    if(!loss_layer.CreateOutputs({&input_data, &label_data, &weight_data}, tmp)) {
      FATAL("Layer will not create outputs!");
    }
    if(!loss_layer.Connect({&input_data, &label_data, &weight_data}, tmp, &net_status)) {
      FATAL("Layer will not connect!");
    }

    loss_layer.OnLayerConnect({}, false);

    if(!Conv::GradientTester::DoGradientTest(&loss_layer, input_data.data, input_data.delta, tmp, epsilon, WriteLossDeltas, CalculateLoss)) {
      LOGERROR << "Failed gradient test!";
    }
  }

  delete[] metadata;
  LOGEND;
}
