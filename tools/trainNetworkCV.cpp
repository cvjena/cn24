/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file trainNetworkCV.cpp
 * \brief Trains a convolutional neural net for KITTI prediction.
 *
 * \author Clemens-Alexander Brust(ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <ctime>

#include <cn24.h>

int main (int argc, char* argv[]) {
  bool DO_TEST = true;
  const unsigned int BATCHSIZE = 48;
  unsigned int TEST_EVERY = 10;
  unsigned int EPOCHS = 40;
  std::string mode = "slow";
  bool RESUME = false;
  std::string resumefile;
  unsigned int RESUME_FROM = 0;
  unsigned int RESUME_ELEMENT = 0;
  unsigned int PATCHSIZE_X = 32;
  unsigned int PATCHSIZE_Y = 32;
  unsigned int SPLIT = 10;

  if (argc < 3) {
    LOGERROR << "USAGE: " << argv[0] << " <DATASET Tensor> <net> [<resume file> <epoch> <cv element>] [fast|mid]";
    LOGEND;
    return -1;
  }
  
  Conv::System::Init();

  int subset = (argv[2][1] == 'X') ? -1 : argv[2][1] - '0';
  if(subset == -1) {
    LOGINFO << "Disabling cross validation run. Using complete training set!";
    LOGINFO << "Adding ten epochs";
    EPOCHS += 10;
    DO_TEST = false;
  } else if((subset > (SPLIT-1)) || (subset < (-1))) {
    FATAL ("Subset number out of bounds: " << subset);
  }

  Conv::Factory* factory = Conv::Factory::getNetFactory (argv[2][0], 49932);
  if (factory == nullptr) {
    FATAL ("Unknown net: " << argv[2]);
  }
  
  PATCHSIZE_X = factory->patchsizex();
  PATCHSIZE_Y = factory->patchsizey();
  
  Conv::TrainerSettings settings = factory->optimal_settings();

  if ( (argc > 3 && std::string (argv[3]).compare ("fast") == 0) ||
       (argc > 6 && std::string (argv[6]).compare ("fast") == 0)) {
    settings.iterations *= 0.01;
    settings.testing_ratio *= 0.01;
    EPOCHS = 5;
    TEST_EVERY = 5;
    DO_TEST = false;
    mode = "fast";
    LOGINFO << "Using fast mode";
  } else if ( (argc > 3 && std::string (argv[3]).compare ("mid") == 0) ||
              (argc > 6 && std::string (argv[6]).compare ("mid") == 0)) {
    settings.iterations *= 0.1;
    settings.testing_ratio *= 0.1;
    EPOCHS = 10;
    mode = "mid";
    LOGINFO << "Using midfast mode";
  }

  if (argc > 5 && std::string (argv[3]).compare ("fast") != 0 &&
      std::string (argv[3]).compare ("mid") != 0) {
    RESUME = true;
    resumefile = std::string (argv[3]);
    RESUME_FROM = atoi (std::string (argv[4]).c_str());
    RESUME_ELEMENT = atoi (std::string (argv[5]).c_str());
    LOGINFO << "Resuming from epoch " << RESUME_FROM << "(" <<
            resumefile << ")";
    LOGINFO << "CV element: " << RESUME_ELEMENT;
  }

  std::ifstream dataset_tensor (argv[1], std::ios::in | std::ios::binary);

  Conv::Tensor training_data;
  Conv::Tensor training_labels;

  training_data.Deserialize (dataset_tensor);
  training_labels.Deserialize (dataset_tensor);

  LOGINFO << "Training: " << training_data << training_labels;

  /*
   * Assemble net
   */
  Conv::S2CVLabeledDataLayer data_layer (training_data, training_labels,
                                         PATCHSIZE_X, PATCHSIZE_Y,
                                         BATCHSIZE, 983882, SPLIT, RESUME_ELEMENT,
                                         true, false, true, false,
                                         Conv::KITTIData::LocalizedError
                                        );
  
  // If subset is -1, S2CVLabeledDataLayer will disable testing and use
  // the complete training set
  data_layer.SetCrossValidationTestingSubset(subset);

  Conv::Net net;

  int data_layer_id = net.AddLayer (&data_layer);
  int output_layer_id =
    factory->AddLayers (net, Conv::Connection (data_layer_id));

  LOGDEBUG << "Output layer id: " << output_layer_id;
  /*
   * Loss functions
   */
  Conv::ErrorLayer loss_layer;
  Conv::BinaryStatLayer binary_stat_layer;

  net.AddLayer (&loss_layer, {
    Conv::Connection (output_layer_id),
    Conv::Connection (data_layer_id, 1),
    Conv::Connection (data_layer_id, 3),
  });
  net.AddLayer (&binary_stat_layer, {
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
    
    std::stringstream ss;
    std::time_t t = std::time (nullptr);
    std::tm tm_ = *std::localtime (&t);

    ss << "snap" << argv[2] << "_" << std::setfill ('0') << std::setw (2)
       << tm_.tm_mday << "." << tm_.tm_mon << "_" << tm_.tm_hour << "."
       << tm_.tm_min << "_" << mode << "_" 
       << "_"
       << (i+1) * TEST_EVERY << ".Tensor";
    std::ofstream outfile ("snapshots/" + ss.str(), std::ios::out | std::ios::binary);
    net.SerializeParameters (outfile);
    LOGDEBUG << "Serialized to " << ss.str();
    
    if (DO_TEST)
      loss = trainer.Test();
    else {
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

  LOGINFO << "Last element: " << data_layer.current_element();
  outfile.close();

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
