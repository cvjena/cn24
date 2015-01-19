/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testMNIST.cpp
 * \brief Builds a feed-forward neural net for MNIST prediction
 *
 * \author Clemens-Alexander Brust(ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>

#include <cn24.h>

int main (int argc, char* argv[]) {
  const unsigned int BATCHSIZE = 50;
  const unsigned int TEST_EVERY = 1;
  const unsigned int EPOCHS = 12;
  
  Conv::TrainerSettings settings;
  settings.learning_rate = 0.3;
  settings.momentum = 0.9;
  settings.l1_weight = 0;
  settings.l2_weight = 0.001;
  settings.epoch_training_ratio = 0.1;
  settings.testing_ratio = 1;
  
  if (argc != 3) {
    LOGERROR << "USAGE: " << argv[0] << " <training tensor> <testing tensor>";
    LOGEND;
    return -1;
  }

  std::ifstream training_file (argv[1], std::ios::in | std::ios::binary);
  std::ifstream testing_file (argv[2], std::ios::in | std::ios::binary);

  Conv::Tensor training_data;
  Conv::Tensor training_labels;
  Conv::Tensor testing_data;
  Conv::Tensor testing_labels;

  training_data.Deserialize (training_file);
  training_labels.Deserialize (training_file);

  testing_data.Deserialize (testing_file);
  testing_labels.Deserialize (testing_file);

  LOGINFO << "Training: " << training_data << training_labels;
  LOGINFO << "Testing: " << testing_data << testing_labels;
  
  /*
   * Flatten tensors
   */
  //training_data.Reshape(60000, 28 * 28, 1, 1);
  //testing_data.Reshape(10000, 28 * 28, 1, 1);

  /*
   * Assemble net
   */
  Conv::SimpleLabeledDataLayer data_layer (training_data, training_labels,
      testing_data, testing_labels, BATCHSIZE);
  Conv::FlattenLayer fl_layer;
  Conv::FullyConnectedLayer fc_layer1 (300);
  Conv::TanhLayer nl_layer1;
  //Conv::FullyConnectedLayer fc_layer2 (150);
  //Conv::TanhLayer nl_layer2;
  Conv::FullyConnectedLayer fc_layer3 (10);
  Conv::SigmoidLayer nl_layer3;
  Conv::ErrorLayer loss_layer;
  Conv::ErrorRateLayer err_layer;

  Conv::Net net;
  
  //fc_layer1.SetLocalLearningRate(1.33);
  fc_layer1.SetBackpropagationEnabled(false);

  int lid_data = net.AddLayer (&data_layer);
  int lid_fl = net.AddLayer(&fl_layer, {Conv::Connection (lid_data) });
  int lid_fc1 = net.AddLayer (&fc_layer1, {Conv::Connection (lid_fl) });
  int lid_nl1 = net.AddLayer (&nl_layer1, {Conv::Connection (lid_fc1) });
  //int lid_fc2 = net.AddLayer (&fc_layer2, {Conv::Connection (lid_nl1) });
  //int lid_nl2 = net.AddLayer (&nl_layer2, {Conv::Connection (lid_fc2) });
  int lid_fc3 = net.AddLayer(&fc_layer3, lid_nl1);
  int lid_nl3 = net.AddLayer(&nl_layer3, lid_fc3);
  net.AddLayer (&loss_layer, {Conv::Connection (lid_nl3),
                              Conv::Connection (lid_data, 1)
                             });
  net.AddLayer (&err_layer, {Conv::Connection (lid_nl3),
                              Conv::Connection (lid_data, 1)
                             });


  net.InitializeWeights();
  
  Conv::Trainer trainer(net, settings);
  
  // Train epochs
  Conv::datum loss = 0;
  
  for(unsigned int i = 0; i < EPOCHS / TEST_EVERY; i++) {
    trainer.Train(TEST_EVERY);
    loss = trainer.Test();
  }
  
  std::ofstream outfile("mnistparams.Tensor", std::ios::out | std::ios::binary);
  net.SerializeParameters(outfile);
  
  outfile.close();
  
  if(loss > 0.05) {
    LOGERROR << "Loss too high!";
    LOGEND;
    return -1;
  }
  
  LOGINFO << "SUCCESS!";
  LOGEND;
  return 0;
}
