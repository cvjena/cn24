/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file classifyImage.cpp
 * @brief Application that uses a pretrained net to segment images.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>

#include <cn24.h>

int main (int argc, char* argv[]) {
  if (argc < 6) {
    LOGERROR << "USAGE: " << argv[0] << " <dataset config file> <net config file> <net parameter tensor> <input image file> <output image file>";
    LOGEND;
    return -1;
  }

  // Capture command line arguments
  std::string output_image_fname (argv[5]);
  std::string input_image_fname (argv[4]);
  std::string param_tensor_fname (argv[3]);
  std::string net_config_fname (argv[2]);
  std::string dataset_config_fname (argv[1]);
  
  // Initialize CN24
  Conv::System::Init();

  // Open network and dataset configuration files
  std::ifstream param_tensor_file(param_tensor_fname,std::ios::in | std::ios::binary);
  std::ifstream net_config_file(net_config_fname,std::ios::in);
  std::ifstream dataset_config_file(dataset_config_fname,std::ios::in);
  
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
  Conv::ConfigurableFactory* factory = new Conv::ConfigurableFactory(net_config_file, 238238, false);
  // Parse dataset configuration file
  Conv::TensorStreamDataset* dataset = Conv::TensorStreamDataset::CreateFromConfiguration(dataset_config_file, true);
  unsigned int CLASSES = dataset->GetClasses();
  
  // Load image
  Conv::Tensor original_data_tensor(input_image_fname);
  
  // Rescale image
  unsigned int width = original_data_tensor.width();
  unsigned int height = original_data_tensor.height();
  if(width & 1)
    width++;
  if(height & 1)
    height++;
  
  Conv::Tensor data_tensor(1, width, height, original_data_tensor.maps());
  data_tensor.Clear();
  Conv::Tensor::CopySample(original_data_tensor, 0, data_tensor, 0);

  // Assemble net
  Conv::Net net;
  Conv::InputLayer input_layer(data_tensor);

  int data_layer_id = net.AddLayer(&input_layer);
  int output_layer_id =
    factory->AddLayers (net, Conv::Connection (data_layer_id), CLASSES);

  // Load network parameters
  net.DeserializeParameters(param_tensor_file);
  
  LOGINFO << "Classifying..." << std::flush;
  net.FeedForward();
  
  Conv::Tensor* net_output_tensor = &net.buffer(output_layer_id)->data;
  Conv::Tensor image_output_tensor(1, net_output_tensor->width(), net_output_tensor->height(), 3);
  
  LOGINFO << "Colorizing..." << std::flush;
  dataset->Colorize(*net_output_tensor, image_output_tensor);
  
  // Recrop image down
  Conv::Tensor small(1, original_data_tensor.width(), original_data_tensor.height(), 3);
  for(unsigned int m = 0; m < 3; m++)
    for(unsigned int y = 0; y < small.height(); y++)
      for(unsigned int x = 0; x < small.width(); x++)
        *small.data_ptr(x,y,m,0) = *image_output_tensor.data_ptr_const(x,y,m,0);

  small.WriteToFile(output_image_fname);

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
