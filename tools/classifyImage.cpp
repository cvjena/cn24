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
  unsigned int original_width = original_data_tensor.width();
  unsigned int original_height = original_data_tensor.height();
  if(width & 1)
    width++;
  if(height & 1)
    height++;
  
  if(width & 2)
    width+=2;
  if(height & 2)
    height+=2;
  
  if(width & 4)
    width+=4;
  if(height & 4)
    height+=4;

  if(width & 8)
    width+=8;
  if(height & 8)
    height+=8;

  if(width & 16)
    width+=16;
  if(height & 16)
    height+=16;
  
  Conv::Tensor data_tensor(1, width, height, original_data_tensor.maps());
  Conv::Tensor helper_tensor(1, width, height, 2);
  data_tensor.Clear();
  helper_tensor.Clear();

  // Copy sample because data_tensor may be slightly larger
  Conv::Tensor::CopySample(original_data_tensor, 0, data_tensor, 0);

  // Initialize helper (spatial prior) tensor

  // Write spatial prior data to helper tensor
  for (unsigned int y = 0; y < original_height; y++) {
    for (unsigned int x = 0; x < original_width; x++) {
      *helper_tensor.data_ptr(x, y, 0, 0) = ((Conv::datum)x) / ((Conv::datum)original_width - 1);
      *helper_tensor.data_ptr(x, y, 1, 0) = ((Conv::datum)y) / ((Conv::datum)original_height - 1);
    }
    for (unsigned int x = original_width; x < width; x++) {
      *helper_tensor.data_ptr(x, y, 0, 0) = 0;
      *helper_tensor.data_ptr(x, y, 1, 0) = 0;
    }
  }
  for (unsigned int y = original_height; y < height; y++) {
    for (unsigned int x = 0; x < height; x++) {
      *helper_tensor.data_ptr(x, y, 0, 0) = 0;
      *helper_tensor.data_ptr(x, y, 1, 0) = 0;
    }
  }
  

  // Assemble net
	Conv::NetGraph graph;
  Conv::InputLayer input_layer(data_tensor, helper_tensor);

	Conv::NetGraphNode input_node(&input_layer);
  input_node.is_input = true;

	graph.AddNode(&input_node);
	bool complete = factory->AddLayers(graph, Conv::NetGraphConnection(&input_node), CLASSES);
	if (!complete)
    FATAL("Failed completeness check, inspect model!");

	graph.Initialize();


  // Load network parameters
  graph.DeserializeParameters(param_tensor_file);
  
  graph.SetIsTesting(true);
  LOGINFO << "Classifying..." << std::flush;
  graph.FeedForward();
  
	Conv::Tensor* net_output_tensor = &graph.GetDefaultOutputNode()->output_buffers[0].combined_tensor->data; // &net.buffer(output_layer_id)->data;
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
