/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testSimpleNet.cpp
 * \brief Test application for a simple neural net with cross validation.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#include <random>
#include <cmath>

#include <cn24.h>

inline Conv::datum f (Conv::datum x1, Conv::datum x2) {
  // We want to approximate this function
  return 1.0 - 2.0 / (exp (2.0 * (-0.3 * x1 + 0.6 * x2 + 0.1)) + 1.0);
}

int main() {
  /*
   * GENERATION
   */
  const int DATAPOINTS = 10000;
  const int BATCHSIZE = 1;
  
  Conv::Tensor data (DATAPOINTS, 2);
  Conv::Tensor labels (DATAPOINTS);

  std::mt19937 randomizer;
  std::uniform_real_distribution<Conv::datum> dist (-3.0, 3.0);

  LOGINFO << "Generating data...";

  for (int i = 0; i < DATAPOINTS; i++) {
    Conv::datum point1 = dist (randomizer);
    Conv::datum point2 = dist (randomizer);
    Conv::datum label = f (point1, point2);
    *data.data_ptr (0, 0, 0, i) = point1;
    *data.data_ptr (1, 0, 0, i) = point2;
    *labels.data_ptr (0, 0, 0, i) = label;
  }
  
  // TODO do this n times for CV
  Conv::CVLabeledDataLayer data_layer (data, labels, BATCHSIZE, 0, 5);
  Conv::FullyConnectedLayer fc_layer1 (1);
  Conv::TanhLayer nl_layer1;
  Conv::ErrorLayer error_layer;

  Conv::Net net;
  int lid_data = net.AddLayer (&data_layer);
  int lid_fc1 = net.AddLayer (&fc_layer1, {Conv::Connection (lid_data) });
  int lid_nl1 = net.AddLayer (&nl_layer1, {Conv::Connection (lid_fc1) });
  int lid_err = net.AddLayer (&error_layer,
  {Conv::Connection (lid_nl1), Conv::Connection (lid_data, 1) });

  LOGINFO << "Training samples: " << data_layer.GetSamplesInTrainingSet() <<
          ", Testing samples: " << data_layer.GetSamplesInTestingSet();
  
  /*
   * TRAINING
   */
  Conv::TrainerSettings settings;
  settings.learning_rate = 0.3;
  settings.l2_weight = 0;
  
  Conv::Trainer trainer(net, settings);
  
  trainer.Train(5);
  Conv::datum testing_loss = trainer.Test();

  if (testing_loss < 0.1) {
    LOGINFO << "SUCCESS";
    LOGEND;
    return 0;
  } else {
    LOGERROR << "LOSS TOO HIGH";
    LOGEND;
    return -1;
  }
}
