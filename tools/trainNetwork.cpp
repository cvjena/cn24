/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
 /**
  * @file trainNetwork.cpp
  * @brief Trains a convolutional neural net for prediction.
  *
  * @author Clemens-Alexander Brust(ikosa dot de at gmail dot com)
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
#include <private/ConfigParsing.h>

bool parseCommand(Conv::Net& net, Conv::Net& testing_net, Conv::Trainer& trainer, Conv::Trainer& testing_trainer, bool hybrid, std::string& command);
void help();

int main(int argc, char* argv[]) {
  bool GRADIENT_CHECK = false;
  bool FROM_SCRIPT = false;
  const bool hybrid = true;
#ifdef LAYERTIME
  const Conv::datum it_factor = 0.01;
#else
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
  Conv::ConfigurableFactory* factory = new Conv::ConfigurableFactory(net_config_file, 8347734);
  factory->InitOptimalSettings();
  LOGDEBUG << "Optimal settings: " << factory->optimal_settings();
  unsigned int BATCHSIZE = factory->optimal_settings().pbatchsize;

  Conv::TrainerSettings settings = factory->optimal_settings();
  settings.epoch_training_ratio = 1 * it_factor;
  settings.testing_ratio = 1 * it_factor;

  // Load dataset
  Conv::TensorStreamDataset* dataset = Conv::TensorStreamDataset::CreateFromConfiguration(dataset_config_file, false, hybrid ? Conv::LOAD_TRAINING_ONLY : Conv::LOAD_BOTH);
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
    
    Conv::Net* testing_net;
    Conv::Trainer* testing_trainer;
    if(hybrid) {
      // Assemble testing net
      Conv::TensorStreamDataset* testing_dataset = Conv::TensorStreamDataset::CreateFromConfiguration(dataset_config_file, false, Conv::LOAD_TESTING_ONLY);
      testing_net = new Conv::Net();
      
      int tdata_layer_id = 0;

      Conv::DatasetInputLayer* tdata_layer = nullptr;
      tdata_layer = new Conv::DatasetInputLayer(*testing_dataset, BATCHSIZE, loss_sampling_p, 983923);
      tdata_layer_id = testing_net->AddLayer(tdata_layer);

      int toutput_layer_id =
        factory->AddLayers(*testing_net, Conv::Connection(tdata_layer_id), CLASSES);

      LOGDEBUG << "Output layer id: " << toutput_layer_id;

      testing_net->AddLayer(factory->CreateLossLayer(CLASSES), {
        Conv::Connection(toutput_layer_id),
        Conv::Connection(tdata_layer_id, 1),
        Conv::Connection(tdata_layer_id, 3),
      });

      // Add appropriate statistics layer
      if (CLASSES == 1) {
        Conv::BinaryStatLayer* tbinary_stat_layer = new Conv::BinaryStatLayer(13, -1, 1);
        testing_net->AddLayer(tbinary_stat_layer, {
          Conv::Connection(toutput_layer_id),
          Conv::Connection(tdata_layer_id, 1),
          Conv::Connection(tdata_layer_id, 3)
        });
      }
      else {
        std::vector<std::string> class_names = dataset->GetClassNames();
        Conv::ConfusionMatrixLayer* tconfusion_matrix_layer = new Conv::ConfusionMatrixLayer(class_names, CLASSES);
        testing_net->AddLayer(tconfusion_matrix_layer, {
          Conv::Connection(toutput_layer_id),
          Conv::Connection(tdata_layer_id, 1),
          Conv::Connection(tdata_layer_id, 3)
        });
      }

      // Shadow training net weights
      std::vector<Conv::CombinedTensor*> training_params;
      std::vector<Conv::CombinedTensor*> testing_params;
      net.GetParameters(training_params);
      testing_net->GetParameters(testing_params);
      
      for(unsigned int p = 0; p < training_params.size(); p++) {
        Conv::CombinedTensor* training_ct = training_params[p];
        Conv::CombinedTensor* testing_ct = testing_params[p];
        testing_ct->data.Shadow(training_ct->data);
        testing_ct->delta.Shadow(training_ct->delta);
      }
          
      testing_trainer = new Conv::Trainer(*testing_net, factory->optimal_settings());
    } else {
      testing_net = &net;
      testing_trainer = &trainer;
    }
    
    
    if(FROM_SCRIPT) {
      LOGINFO << "Executing script: " << script_fname;
      std::ifstream script_file(script_fname, std::ios::in);
      if(!script_file.good()) {
        FATAL("Cannot open " << script_fname);
      }
      while (true) {
        std::string command;
        std::getline(script_file, command);
        if (!parseCommand(net, *testing_net, trainer, *testing_trainer, hybrid, command) || script_file.eof())
          break;
      }
    } else {
      LOGINFO << "Enter \"help\" for information on how to use this program";
      while (true) {
        std::cout << "\n > " << std::flush;
        std::string command;
        std::getline(std::cin, command);
        if (!parseCommand(net, *testing_net, trainer, *testing_trainer, hybrid, command))
          break;
      }
    }
  }

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}


bool parseCommand(Conv::Net& net, Conv::Net& testing_net, Conv::Trainer& trainer, Conv::Trainer& testing_trainer, bool hybrid, std::string& command) {
  if (command.compare("q") == 0 || command.compare("quit") == 0) {
    return false;
  }
  else if (command.compare(0, 5, "train") == 0) {
    unsigned int epochs = 1;
    Conv::ParseCountIfPossible(command, "epochs", epochs);
    trainer.Train(epochs);
    LOGINFO << "Training complete.";
  }
  else if (command.compare(0, 4, "test") == 0) {
    unsigned int layerview = 0;
    Conv::ParseCountIfPossible(command, "view", layerview);
    testing_net.SetLayerViewEnabled(layerview == 1);
    testing_trainer.Test();
    testing_net.SetLayerViewEnabled(false);
    LOGINFO << "Testing complete.";
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
        LOGINFO << "Loaded parameters from " << param_file_name;
        if(hybrid) {
          LOGDEBUG << "Reshadowing tensors...";
          // Shadow training net weights
          std::vector<Conv::CombinedTensor*> training_params;
          std::vector<Conv::CombinedTensor*> testing_params;
          net.GetParameters(training_params);
          testing_net.GetParameters(testing_params);
          
          for(unsigned int p = 0; p < training_params.size(); p++) {
            Conv::CombinedTensor* training_ct = training_params[p];
            Conv::CombinedTensor* testing_ct = testing_params[p];
            testing_ct->data.Shadow(training_ct->data);
            testing_ct->delta.Shadow(training_ct->delta);
          } 
        }
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
        LOGINFO << "Written parameters to " << param_file_name;
      }
      else {
        LOGERROR << "Cannot open " << param_file_name;
      }
      param_file.close();
    }
  }
  else if (command.compare(0, 9, "set epoch") == 0) {
    unsigned int epoch = 0;
    Conv::ParseCountIfPossible(command, "epoch", epoch);
    LOGINFO << "Setting current epoch to " << epoch;
    trainer.SetEpoch(epoch);
  }
  else if(command.compare(0, 5, "reset") == 0) {
    LOGINFO << "Resetting parameters";
    net.InitializeWeights();
  }
  else if (command.compare(0, 4, "help") == 0) {
    help();
  } else {
    LOGWARN << "Unknown command: " << command;
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
    << "  set epoch=<epoch>\n"
    << "    Sets the current epoch\n\n"
    << "  reset\n"
    << "    Reinitializes the nets parameters\n\n"
    << "  load file=<path> [last_layer=<l>]\n"
    << "    Load parameters from a file for all layers up to l (default: all layers)\n\n"
    << "  save file=<path>\n"
    << "    Save parameters to a file\n";
}
