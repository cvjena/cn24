/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file trainNetwork.cpp
 * \brief Trains a convolutional neural net for prediction.
 *
 * \author Clemens-A. Brust(ikosa.de@gmail.com)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <iomanip>
#include <ctime>
#include <cstring>

#include <cn24.h>

int main (int argc, char* argv[]) {
  bool DO_TEST = true;
#ifdef LAYERTIME
  const unsigned int BATCHSIZE = 1000;
  unsigned int TEST_EVERY = 1;
  const Conv::datum it_factor = 0.01;
#else
  const unsigned int BATCHSIZE = 2;
  unsigned int TEST_EVERY = 5;
  const Conv::datum it_factor = 1;
#endif
  unsigned int EPOCHS = 120;
  std::string mode = "slow";

  if (argc < 3) {
    LOGERROR << "USAGE: " << argv[0] << " <dataset config file> <net config file>";
    LOGEND;
    return -1;
  }
  
  Conv::System::Init();

  //Conv::Factory* factory = Conv::Factory::getNetFactory (argv[3][0], 49932);
  std::string net_config_fname (argv[2]);
  std::string dataset_config_fname (argv[1]);
  
  std::ifstream net_config_file(net_config_fname,std::ios::in);
  std::ifstream dataset_config_file(dataset_config_fname,std::ios::in);
  
  if(!net_config_file.good()) {
    FATAL("Cannot open net configuration file!");
  }
  net_config_fname = net_config_fname.substr(net_config_fname.rfind("/")+1);
  
  Conv::Factory* factory = new Conv::ConfigurableFactory(net_config_file, Conv::FCN);
  factory->InitOptimalSettings();
  
  Conv::TrainerSettings settings = factory->optimal_settings();
  settings.epoch_training_ratio = 1 * it_factor;
  settings.testing_ratio = 1 * it_factor;

  Conv::TensorStreamDataset* dataset = Conv::TensorStreamDataset::CreateFromConfiguration(dataset_config_file);
  unsigned int CLASSES = dataset->GetClasses();
  
  /*
   * Assemble net
   */
  Conv::DatasetInputLayer data_layer(*dataset, BATCHSIZE, 983923);
  Conv::Net net;

  int data_layer_id = net.AddLayer (&data_layer);
  int output_layer_id =
    factory->AddLayers (net, Conv::Connection (data_layer_id), CLASSES);

  LOGDEBUG << "Output layer id: " << output_layer_id;

  net.AddLayer (factory->CreateLossLayer(CLASSES), {
    Conv::Connection (output_layer_id),
    Conv::Connection (data_layer_id, 1),
    Conv::Connection (data_layer_id, 3),
  });

  if(CLASSES == 1) {
    Conv::BinaryStatLayer* binary_stat_layer = new Conv::BinaryStatLayer(13,-1,1);
    net.AddLayer (binary_stat_layer, {
      Conv::Connection (output_layer_id),
      Conv::Connection (data_layer_id, 1),
      Conv::Connection (data_layer_id, 3)
    });
  } else {
    std::vector<std::string> class_names = dataset->GetClassNames();
    Conv::ConfusionMatrixLayer* confusion_matrix_layer = new Conv::ConfusionMatrixLayer(class_names, CLASSES);
    net.AddLayer (confusion_matrix_layer, {
      Conv::Connection (output_layer_id),
      Conv::Connection (data_layer_id, 1),
      Conv::Connection (data_layer_id, 3)
    });
  }

  Conv::Trainer trainer (net, settings);
  net.InitializeWeights();

  for (unsigned int i = 0; i < EPOCHS / TEST_EVERY; i++) {
    trainer.Train (TEST_EVERY);
#ifdef LAYERTIME
    net.PrintAndResetLayerTime(settings.iterations * settings.epoch_training_ratio * BATCHSIZE);
#endif
    
    std::stringstream ss;
    std::time_t t = std::time (nullptr);
    std::tm tm_ = *std::localtime (&t);

    ss << "snap" << net_config_fname << "_" << std::setfill ('0') << std::setw (2)
       << tm_.tm_mday << "." << tm_.tm_mon << "_" << tm_.tm_hour << "."
       << tm_.tm_min << "_" << mode << "_" 
       << "_"
       << (i+1) * TEST_EVERY << ".Tensor";
    std::ofstream outfile ("snapshots/" + ss.str(), std::ios::out | std::ios::binary);
    net.SerializeParameters (outfile);
    LOGDEBUG << "Serialized to " << ss.str();
    
    if (DO_TEST) {
      net.SetLayerViewEnabled(true);
      trainer.Test();
      net.SetLayerViewEnabled(false);
#ifdef LAYERTIME
      net.PrintAndResetLayerTime(settings.iterations * settings.testing_ratio * BATCHSIZE);
#endif
    } else {
      LOGDEBUG << "Skipping test...";
    }
  }

  std::stringstream ss;
  std::time_t t = std::time (nullptr);
  std::tm tm_ = *std::localtime (&t);

  ss << "n" << net_config_fname << "_" << std::setfill ('0') << std::setw (2)
     << tm_.tm_mday << "." << tm_.tm_mon << "_" << tm_.tm_hour << "."
     << tm_.tm_min << "_" << mode << "_" << ".Tensor";
  std::ofstream outfile (ss.str(), std::ios::out | std::ios::binary);
  net.SerializeParameters (outfile);
  
  ss.str("");
  ss << "logs/n" << dataset_config_fname << "_" << std::setfill ('0') << std::setw (2)
     << tm_.tm_mday << "." << tm_.tm_mon << "_" << tm_.tm_hour << "."
     << tm_.tm_min << "_" << net_config_fname << "_" << ".log";

  LOGINFO << "Last element: " << data_layer.current_element();
  outfile.close();

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
