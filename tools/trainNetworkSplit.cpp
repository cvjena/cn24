/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file trainNetworkSplit.cpp
 * \brief Trains a convolutional neural net for LabelMeFacade prediction.
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
  const unsigned int BATCHSIZE = 48;
  unsigned int TEST_EVERY = 20;
  const Conv::datum it_factor = 1;
#endif
  unsigned int EPOCHS = 120;
  std::string mode = "slow";
  bool RESUME = false;
  bool EQUAL_WEIGHTS = false;
  std::string resumefile;
  unsigned int RESUME_FROM = 0;
  unsigned int RESUME_ELEMENT = 0;
  unsigned int PATCHSIZE_X = 32;
  unsigned int PATCHSIZE_Y = 32;
  unsigned int SPLIT = 10;

  if (argc < 4) {
    LOGERROR << "USAGE: " << argv[0] << " <training tensor> <testing tensor> <net> [w]";
    LOGEND;
    return -1;
  }
  
  Conv::System::Init();

  Conv::Factory* factory = Conv::Factory::getNetFactory (argv[3][0], 49932);
  if (factory == nullptr) {
    FATAL ("Unknown net: " << argv[3]);
  }
  
  PATCHSIZE_X = factory->patchsizex();
  PATCHSIZE_Y = factory->patchsizey();
  
  if(!strcmp(argv[2], "no_test")) {
    LOGINFO << "Disabling test";
    DO_TEST = false;
  }
  
  if(argc > 4) {
    if(argv[4][0] == 'w') {
      EQUAL_WEIGHTS = true;
      LOGINFO << "Setting weights to compensate for class probabilities...";
    }
  }
  
  Conv::TrainerSettings settings = factory->optimal_settings();
  settings.epoch_training_ratio = 1 * it_factor;
  settings.testing_ratio = 1 * it_factor;

  std::ifstream training_stream (argv[1], std::ios::in | std::ios::binary);
  std::ifstream testing_stream (argv[2], std::ios::in | std::ios::binary);
  
  // TODO read this from somewhere
  std::vector<std::string> class_names {
    "various",
    "building",
    "car",
    "door",
    "pavement",
    "road",
    "sky",
    "vegetation",
    "window"
  };
  
  Conv::datum class_weights[]  {
    0.0,	// various
    0.2683453,	// building
    3.047443,	// car
    13.55019,	// door
    1.914624,	// pavement
    0.74853,	// road
    0.9597495,	// sky
    1.522002,	// vegetation
    3.181684	// window
  };
  Conv::datum sum = 0;
  for(unsigned int c = 0; c < 9; c++) {
    sum += class_weights[c];
  }
  for(unsigned int c = 0; c < 9; c++) {
    class_weights[c] /= (sum / 8.0);
  }

  /*
   * Assemble net
   */
  Conv::S2SplitLabeledDataLayer data_layer (training_stream, testing_stream,
                                         PATCHSIZE_X, PATCHSIZE_Y,
                                         BATCHSIZE, 983882, 0, true, false,
                                         false, 0, Conv::DefaultLocalizedErrorFunction,
					 EQUAL_WEIGHTS ? class_weights : nullptr);
  
  Conv::Net net;

  int data_layer_id = net.AddLayer (&data_layer);
  int output_layer_id =
    factory->AddLayers (net, Conv::Connection (data_layer_id), 9);

  LOGDEBUG << "Output layer id: " << output_layer_id;
  /*
   * Loss functions
   */
  //Conv::MultiClassErrorLayer loss_layer(9);
  Conv::ConfusionMatrixLayer confusion_matrix_layer(class_names, 9);

  net.AddLayer (factory->CreateLossLayer(9), {
    Conv::Connection (output_layer_id),
    Conv::Connection (data_layer_id, 1),
    Conv::Connection (data_layer_id, 3),
  });
  net.AddLayer (&confusion_matrix_layer, {
    Conv::Connection (output_layer_id),
    Conv::Connection (data_layer_id, 1),
    Conv::Connection (data_layer_id, 3)
  });

  Conv::Trainer trainer (net, settings);
  if (RESUME) {
    trainer.SetEpoch (RESUME_FROM);
    std::ifstream param_stream (resumefile, std::ios::in | std::ios::binary);
    if (param_stream.good())
      net.DeserializeParameters (param_stream);
    else {
      FATAL ("Cannot open " << resumefile);
    }
  } else
    net.InitializeWeights();

  // Train epochs
  Conv::datum loss = 0;

  for (unsigned int i = 0; i < EPOCHS / TEST_EVERY; i++) {
    trainer.Train (TEST_EVERY);
#ifdef LAYERTIME
    net.PrintAndResetLayerTime(settings.iterations * settings.epoch_training_ratio * BATCHSIZE);
#endif
    
    std::stringstream ss;
    std::time_t t = std::time (nullptr);
    std::tm tm_ = *std::localtime (&t);

    ss << "snap" << argv[3] << "_" << std::setfill ('0') << std::setw (2)
       << tm_.tm_mday << "." << tm_.tm_mon << "_" << tm_.tm_hour << "."
       << tm_.tm_min << "_" << mode << "_" 
       << "_"
       << (i+1) * TEST_EVERY << ".Tensor";
    std::ofstream outfile ("snapshots/" + ss.str(), std::ios::out | std::ios::binary);
    net.SerializeParameters (outfile);
    LOGDEBUG << "Serialized to " << ss.str();
    
    if (DO_TEST) {
      loss = trainer.Test();
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

  ss << "n" << argv[2] << "_" << std::setfill ('0') << std::setw (2)
     << tm_.tm_mday << "." << tm_.tm_mon << "_" << tm_.tm_hour << "."
     << tm_.tm_min << "_" << mode << "_" << ".Tensor";
  std::ofstream outfile (ss.str(), std::ios::out | std::ios::binary);
  net.SerializeParameters (outfile);
  
  ss.str("");
  ss << "logs/n" << argv[2] << "_" << std::setfill ('0') << std::setw (2)
     << tm_.tm_mday << "." << tm_.tm_mon << "_" << tm_.tm_hour << "."
     << tm_.tm_min << "_" << argv[3] << "_" << ".log";
  std::ofstream csvfile (ss.str(), std::ios::out | std::ios::binary);
  confusion_matrix_layer.PrintCSV(csvfile);

  LOGINFO << "Last element: " << data_layer.current_element();
  outfile.close();
  csvfile.close();

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
