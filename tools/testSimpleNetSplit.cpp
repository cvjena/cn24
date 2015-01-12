/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testSimpleNetSplit.cpp
 * \brief Test application for a simple neural net with split data set.
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
  const int DATAPOINTS_TRAINING = 9000;
  const int DATAPOINTS_TEST = 1000;
  const int BATCHSIZE = 1;
  
  Conv::Tensor training_data (DATAPOINTS_TRAINING, 2);
  Conv::Tensor training_labels (DATAPOINTS_TRAINING);
  Conv::Tensor testing_data (DATAPOINTS_TEST, 2);
  Conv::Tensor testing_labels (DATAPOINTS_TEST);

  std::mt19937 randomizer;
  std::uniform_real_distribution<Conv::datum> dist (-3.0, 3.0);

  LOGINFO << "Generating data...";
  for (int i = 0; i < DATAPOINTS_TRAINING; i++) {
    Conv::datum point1 = dist (randomizer);
    Conv::datum point2 = dist (randomizer);
    Conv::datum label = f (point1, point2);
    *training_data.data_ptr (0, 0, 0, i) = point1;
    *training_data.data_ptr (1, 0, 0, i) = point2;
    *training_labels.data_ptr (0, 0, 0, i) = label;
  }
  
  for (int i = 0; i < DATAPOINTS_TEST; i++) {
    Conv::datum point1 = dist (randomizer);
    Conv::datum point2 = dist (randomizer);
    Conv::datum label = f (point1, point2);
    *testing_data.data_ptr (0, 0, 0, i) = point1;
    *testing_data.data_ptr (1, 0, 0, i) = point2;
    *testing_labels.data_ptr (0, 0, 0, i) = label;
  }
  
  Conv::SimpleLabeledDataLayer data_layer (training_data, training_labels,
                                           testing_data, testing_labels,
                                           BATCHSIZE );
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
