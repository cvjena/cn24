/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file classifyImage.cpp
 * \brief Application that uses a pretrained net to segment images.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>

#include <cn24.h>

int main (int argc, char* argv[]) {
  unsigned int BATCHSIZE = 1;
  if (argc < 6) {
    LOGERROR << "USAGE: " << argv[0] << "<dataset config file> <net config file> <net parameter tensor> <input image file> <output image file>";
    LOGEND;
    return -1;
  }

  std::string output_image_fname (argv[5]);
  if(output_image_fname.length() < 7 || output_image_fname.compare(output_image_fname.length()-6,6,"Tensor") != 0) {
    LOGWARN << "NOTE: This utility writes binary tensor files!";
  }
  std::string input_image_fname (argv[4]);
  std::string param_tensor_fname (argv[3]);
  std::string net_config_fname (argv[2]);
  std::string dataset_config_fname (argv[1]);
  
  Conv::System::Init();

  // Open network and dataset configuration files
  std::ofstream output_image_file(output_image_fname,std::ios::out|std::ios::binary);
  std::ifstream input_image_file(input_image_fname,std::ios::in | std::ios::binary);
  std::ifstream param_tensor_file(param_tensor_fname,std::ios::in | std::ios::binary);
  std::ifstream net_config_file(net_config_fname,std::ios::in);
  std::ifstream dataset_config_file(dataset_config_fname,std::ios::in);
  
  if(!output_image_file.good()) {
    FATAL("Cannot open output tensor file!");
  }
  if(!param_tensor_file.good()) {
    FATAL("Cannot open param tensor file!");
  }
  if(!net_config_file.good()) {
    FATAL("Cannot open net configuration file!");
  }
  if(!dataset_config_file.good()) {
    FATAL("Cannot open dataset configuration file!");
  }
  
  // Parse network configuration file
  Conv::Factory* factory = new Conv::ConfigurableFactory(net_config_file, Conv::FCN);
  factory->InitOptimalSettings();
  LOGDEBUG << "Optimal settings: " << factory->optimal_settings();
  
  Conv::TensorStreamDataset* dataset = Conv::TensorStreamDataset::CreateFromConfiguration(dataset_config_file, true);
  unsigned int CLASSES = dataset->GetClasses();
  
  // Load image
  Conv::Tensor data_tensor;
  Conv::Tensor helper_tensor;
#ifdef BUILD_PNG
  if((input_image_fname.compare(input_image_fname.length() - 3, 3, "png") == 0)
    || (input_image_fname.compare(input_image_fname.length() - 3, 3, "PNG") == 0)
  ) {
    Conv::PNGLoader::LoadFromStream(input_image_file, data_tensor);
  }
#endif
#ifdef BUILD_JPG
  if((input_image_fname.compare(input_image_fname.length() - 3, 3, "jpg") == 0)
    || (input_image_fname.compare(input_image_fname.length() - 3, 3, "jpeg") == 0)
    || (input_image_fname.compare(input_image_fname.length() - 3, 3, "JPG") == 0)
    || (input_image_fname.compare(input_image_fname.length() - 3, 3, "JPEG") == 0)
  ) {
    input_image_file.close();
    Conv::JPGLoader::LoadFromFile(input_image_fname, data_tensor);
  }
#endif
  helper_tensor.Resize(1, data_tensor.width(), data_tensor.height(), 2);
  
  // Assemble net
  Conv::Net net;
  Conv::InputLayer input_layer(data_tensor, helper_tensor);

  int data_layer_id = net.AddLayer(&input_layer);
  int output_layer_id =
    factory->AddLayers (net, Conv::Connection (data_layer_id), CLASSES);

  LOGDEBUG << "Output layer id: " << output_layer_id;

  // net.InitializeWeights();
  net.DeserializeParameters(param_tensor_file);
  
  LOGINFO << "Classifying..." << std::flush;
  net.FeedForward();
  
  Conv::Tensor* net_output_tensor = &net.buffer(output_layer_id)->data;
  Conv::Tensor image_output_tensor(1, net_output_tensor->width(), net_output_tensor->height(), 3);
  
  LOGINFO << "Colorizing..." << std::flush;
  dataset->Colorize(*net_output_tensor, image_output_tensor);
  image_output_tensor.Serialize(output_image_file);

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
