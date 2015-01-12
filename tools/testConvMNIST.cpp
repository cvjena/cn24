/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testConvMNIST.cpp
 * \brief Builds a convolutional neural net for MNIST prediction
 *
 * \author Clemens-A. Brust(ikosa.de@gmail.com)
 */

#include <iostream>
#include <fstream>

#include <cn24.h>

int main (int argc, char* argv[]) {
  const unsigned int BATCHSIZE = 24;
  const unsigned int TEST_EVERY = 5;
  const unsigned int EPOCHS = 10;

  Conv::TrainerSettings settings;
  settings.learning_rate = 0.008;
  settings.momentum = 0.9;
  settings.gamma = 0.0001;
  settings.exponent = 0.75;
  settings.l1_weight = 0;
  settings.l2_weight = 0.0005;
  settings.epoch_training_ratio = 0.0166667;
  settings.testing_ratio = 1;

  if (argc < 3 || argc > 5) {
    LOGERROR << "USAGE: " << argv[0] <<
             " <training tensor> <testing tensor> [fast] [<net params> <epoch>]";
    LOGEND;
    return -1;
  }
  
  if (argc > 3 && std::string(argv[3]).compare("fast") == 0) {
    settings.epoch_training_ratio *= 0.1;
    settings.testing_ratio *= 0.1;
    LOGINFO << "Using fast mode";
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
   * Assemble net
   */
  Conv::SimpleLabeledDataLayer data_layer (training_data, training_labels,
      testing_data, testing_labels, BATCHSIZE);

  /*
   * Hidden layers
   */
  Conv::ConvolutionLayer layer1 (5, 5, 20, 30383920);      // 28x28 (after border addition)
  Conv::MaxPoolingLayer layer2 (2, 2);          // 14x14
  //Conv::ReLULayer layer3;                       // 14x14
  Conv::ConvolutionLayer layer4 (5, 5, 50, 847202);     // 10x10x16
  Conv::MaxPoolingLayer layer5 (2, 2);          // 5x5x16
  //Conv::ReLULayer layer6;                       // 5x5x16
  Conv::FlattenLayer layer7;                    // 400
  Conv::FullyConnectedLayer layer8 (500, 7840289);        // 80
  Conv::ReLULayer layer9;
  //Conv::FullyConnectedLayer layer10 (120);      // 120
  //Conv::TanhLayer layer11;
  Conv::FullyConnectedLayer layer12 (10, 89347);       // 10
  Conv::SigmoidLayer layer13;

  /*
   * Loss functions
   */
  Conv::ErrorLayer loss_layer;
  Conv::ErrorRateLayer err_layer;
  Conv::AccuracyLayer acc_layer;

  Conv::Net net;

  /*
   * Individual learning rates
   */
  layer1.SetBackpropagationEnabled (false);
  //layer12.SetBackpropagationEnabled(false);
  // layer1.SetLocalLearningRate (4.0);
  //layer4.SetLocalLearningRate (1.33);
  /*layer8.SetLocalLearningRate (0.7);
  layer10.SetLocalLearningRate (0.7);*/
  //layer12.SetLocalLearningRate (0.7);

  /*
   * Add layers to net
   */
  int data_layer_id = net.AddLayer (&data_layer);
  int layer1_id = net.AddLayer (&layer1, data_layer_id);
  int layer2_id = net.AddLayer (&layer2, layer1_id);
  // int layer3_id = net.AddLayer (&layer3, layer2_id);
  int layer4_id = net.AddLayer (&layer4, layer2_id);
  int layer5_id = net.AddLayer (&layer5, layer4_id);
  // int layer6_id = net.AddLayer (&layer6, layer5_id);
  int layer7_id = net.AddLayer (&layer7, layer5_id);
  int layer8_id = net.AddLayer (&layer8, layer7_id);
  int layer9_id = net.AddLayer (&layer9, layer8_id);
  //int layer10_id = net.AddLayer (&layer10, layer9_id);
  //int layer11_id = net.AddLayer (&layer11, layer10_id);
  int layer12_id = net.AddLayer (&layer12, layer9_id);
  int layer13_id = net.AddLayer (&layer13, layer12_id);

  const int output_layer_id = layer13_id;

  net.AddLayer (&loss_layer, {
    Conv::Connection (output_layer_id),
    Conv::Connection (data_layer_id, 1),
  });
  net.AddLayer (&err_layer, {
    Conv::Connection (output_layer_id),
    Conv::Connection (data_layer_id, 1)
  });
  net.AddLayer (&acc_layer, {
    Conv::Connection (output_layer_id),
    Conv::Connection (data_layer_id, 1)
  });

  Conv::Trainer trainer (net, settings);

  if (argc == 5 && std::string(argv[4]).compare("fast") != 0) {
    std::ifstream parameter_file
    (std::string (argv[3]), std::ios::in | std::ios::binary);

    LOGINFO << "Reading " << argv[3] << "...";
    net.DeserializeParameters (parameter_file);

    std::string s_epoch (argv[4]);
    unsigned int epoch = atoi (s_epoch.c_str());

    LOGINFO << "Rewinding from epoch " << epoch;
    trainer.SetEpoch(epoch);
  } else {
    net.InitializeWeights();
  }

  // Train epochs
  Conv::datum loss = 0;

  for (unsigned int i = 0; i < EPOCHS / TEST_EVERY; i++) {
    trainer.Train (TEST_EVERY);
    loss = trainer.Test();
  }

  std::ofstream outfile ("mnistparams.Tensor", std::ios::out | std::ios::binary);
  net.SerializeParameters (outfile);

  outfile.close();

  if (loss > 0.02) {
    LOGERROR << "Loss too high!";
    LOGEND;
    return -1;
  }

  LOGINFO << "SUCCESS!";
  LOGEND;
  return 0;
}
