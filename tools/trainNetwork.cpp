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
  * \author Clemens-Alexander Brust(ikosa dot de at gmail dot com)
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

bool parseCommand(Conv::Net& net, Conv::Trainer& trainer, std::string& command);
void help();

int main(int argc, char* argv[]) {
  bool DO_TEST = true;
  bool GRADIENT_CHECK = false;
  bool FROM_SCRIPT = false;
#ifdef LAYERTIME
  const unsigned int BATCHSIZE = 1000;
  unsigned int TEST_EVERY = 1;
  const Conv::datum it_factor = 0.01;
#else
  unsigned int BATCHSIZE = 4;
  const Conv::datum it_factor = 1;
  const Conv::datum loss_sampling_p = 0.25;
#endif

  if (argc < 3) {
    LOGERROR << "USAGE: " << argv[0] << " <dataset config file> <net config file> {[script file]|gradient_check}";
    LOGEND;
    return -1;
  }

  std::string script_fname;
  if (argc > 3 && std::string(argv[3]).compare("gradient_check") == 0) {
    GRADIENT_CHECK = true;
  } else if(argc > 3) {
    FROM_SCRIPT = true;
    script_fname = argv[3];
  }

  std::string net_config_fname(argv[2]);
  std::string dataset_config_fname(argv[1]);

  Conv::System::Init();

  // Open network and dataset configuration files
  std::ifstream net_config_file(net_config_fname, std::ios::in);
  std::ifstream dataset_config_file(dataset_config_fname, std::ios::in);

  if (!net_config_file.good()) {
    FATAL("Cannot open net configuration file!");
  }

  net_config_fname = net_config_fname.substr(net_config_fname.rfind("/") + 1);

  if (!dataset_config_file.good()) {
    FATAL("Cannot open dataset configuration file!");
  }

  dataset_config_fname = dataset_config_fname.substr(net_config_fname.rfind("/") + 1);

  // Parse network configuration file
  Conv::Factory* factory = new Conv::ConfigurableFactory(net_config_file, Conv::FCN);
  factory->InitOptimalSettings();
  LOGDEBUG << "Optimal settings: " << factory->optimal_settings();

  Conv::TrainerSettings settings = factory->optimal_settings();
  settings.epoch_training_ratio = 1 * it_factor;
  settings.testing_ratio = 1 * it_factor;

  // Load dataset
  Conv::TensorStreamDataset* dataset = Conv::TensorStreamDataset::CreateFromConfiguration(dataset_config_file);
  unsigned int CLASSES = dataset->GetClasses();

  // Assemble net
  Conv::Net net;
  int data_layer_id = 0;

  Conv::DatasetInputLayer* data_layer = nullptr;
  if (GRADIENT_CHECK) {
    Conv::Tensor* data_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), dataset->GetInputMaps());
    Conv::Tensor* weight_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), 1);
    Conv::Tensor* label_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), dataset->GetLabelMaps());
    Conv::Tensor* helper_tensor = new Conv::Tensor(BATCHSIZE, dataset->GetWidth(), dataset->GetHeight(), 2);
    for (unsigned int b = 0; b < BATCHSIZE; b++)
      dataset->GetTestingSample(*data_tensor, *label_tensor, *weight_tensor, b, b);
    Conv::InputLayer* input_layer = new Conv::InputLayer(*data_tensor, *label_tensor, *helper_tensor, *weight_tensor);
    data_layer_id = net.AddLayer(input_layer);
  }
  else {
    data_layer = new Conv::DatasetInputLayer(*dataset, BATCHSIZE, loss_sampling_p, 983923);
    data_layer_id = net.AddLayer(data_layer);
  }

  int output_layer_id =
    factory->AddLayers(net, Conv::Connection(data_layer_id), CLASSES);

  LOGDEBUG << "Output layer id: " << output_layer_id;

  net.AddLayer(factory->CreateLossLayer(CLASSES), {
    Conv::Connection(output_layer_id),
    Conv::Connection(data_layer_id, 1),
    Conv::Connection(data_layer_id, 3),
  });

  // Add appropriate statistics layer
  if (CLASSES == 1) {
    Conv::BinaryStatLayer* binary_stat_layer = new Conv::BinaryStatLayer(13, -1, 1);
    net.AddLayer(binary_stat_layer, {
      Conv::Connection(output_layer_id),
      Conv::Connection(data_layer_id, 1),
      Conv::Connection(data_layer_id, 3)
    });
  }
  else {
    std::vector<std::string> class_names = dataset->GetClassNames();
    Conv::ConfusionMatrixLayer* confusion_matrix_layer = new Conv::ConfusionMatrixLayer(class_names, CLASSES);
    net.AddLayer(confusion_matrix_layer, {
      Conv::Connection(output_layer_id),
      Conv::Connection(data_layer_id, 1),
      Conv::Connection(data_layer_id, 3)
    });
  }

  // Initialize net with random weights
  net.InitializeWeights();

  if (GRADIENT_CHECK) {
    Conv::GradientTester::TestGradient(net);
  }
  else {
    Conv::Trainer trainer(net, settings);
    if(FROM_SCRIPT) {
      LOGINFO << "Executing script: " << script_fname;
      std::ifstream script_file(script_fname, std::ios::in);
      if(!script_file.good()) {
        FATAL("Cannot open " << script_fname);
      }
      while (true) {
        std::string command;
        std::getline(script_file, command);
        if (!parseCommand(net, trainer, command) || script_file.eof())
          break;
      }
    } else {
      LOGINFO << "Enter \"help\" for information on how to use this program";
      while (true) {
        std::cout << "\n > " << std::flush;
        std::string command;
        std::getline(std::cin, command);
        if (!parseCommand(net, trainer, command))
          break;
      }
    }
  }

  /*
  {
    Conv::Trainer trainer ( net, settings );
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

    LOGINFO << "Last element: " << data_layer->current_element();
    outfile.close();
  }
  */

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}


bool parseCommand(Conv::Net& net, Conv::Trainer& trainer, std::string& command) {
  if (command.compare("q") == 0 || command.compare("quit") == 0) {
    return false;
  }
  else if (command.compare(0, 5, "train") == 0) {
    unsigned int epochs = 1;
    Conv::ParseCountIfPossible(command, "epochs", epochs);
    trainer.Train(epochs);
  }
  else if (command.compare(0, 4, "test") == 0) {
    unsigned int layerview = 0;
    Conv::ParseCountIfPossible(command, "view", layerview);
    net.SetLayerViewEnabled(layerview == 1);
    trainer.Test();
    net.SetLayerViewEnabled(false);
  }
  else if (command.compare(0, 4, "load") == 0) {
    std::string param_file_name;
    unsigned int last_layer = 0;
    Conv::ParseStringParamIfPossible(command, "file", param_file_name);
    Conv::ParseCountIfPossible(command, "last_layer", last_layer);
    if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    }
    else {
      std::ifstream param_file(param_file_name, std::ios::in | std::ios::binary);
      if (param_file.good()) {
        net.DeserializeParameters(param_file, last_layer);
      }
      else {
        LOGERROR << "Cannot open " << param_file_name;
      }
      param_file.close();
    }
  }
  else if (command.compare(0, 4, "save") == 0) {
    std::string param_file_name;
    Conv::ParseStringParamIfPossible(command, "file", param_file_name);
    if (param_file_name.length() == 0) {
      LOGERROR << "Filename needed!";
    }
    else {
      std::ofstream param_file(param_file_name, std::ios::out | std::ios::binary);
      if (param_file.good()) {
        net.SerializeParameters(param_file);
      }
      else {
        LOGERROR << "Cannot open " << param_file_name;
      }
      param_file.close();
    }
  }
  else if (command.compare(0, 4, "help") == 0) {
    help();
  }

  return true;
}

void help() {
  std::cout << "You can use the following commands:\n";
  std::cout
    << "  train [epochs=<n>]\n"
    << "    Train the network for n epochs (default: 1)\n\n"
    << "  test\n"
    << "    Test the network\n\n"
    << "  load file=<path> [last_layer=<l>]\n"
    << "    Load parameters from a file for all layers up to l (default: all layers)\n\n"
    << "  save file=<path>\n"
    << "    Save parameters to a file\n";
}
