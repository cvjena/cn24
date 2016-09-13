/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file predictImage.cpp
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
  if (argc < 5) {
    LOGERROR << "USAGE: " << argv[0] << " <dataset config file> <net config file> <net parameter tensor> <input image file> [<output image file>]";
    LOGEND;
    return -1;
  }

  // Capture command line arguments
  std::string output_image_fname;
  if(argc > 5)
    output_image_fname = argv[5];
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
  Conv::JSON net_json = Conv::JSON::parse(net_config_file);
  net_json["net"]["error_layer"] = "no";

  Conv::JSONNetGraphFactory* factory = new Conv::JSONNetGraphFactory(net_json);

  // Parse dataset configuration file
  Conv::JSON dataset_json = Conv::JSON::parse(dataset_config_file);
  // Remove actual data to avoid loading times
  dataset_json["data"] = Conv::JSON::array();

  Conv::ClassManager class_manager;
  Conv::Dataset* dataset = Conv::JSONDatasetFactory::ConstructDataset(dataset_json, &class_manager);
  unsigned int CLASSES = dataset->GetClasses();

  // Load image
  Conv::Tensor original_data_tensor(input_image_fname);

  if(dataset->GetTask() == Conv::SEMANTIC_SEGMENTATION) {

    // Rescale image
    unsigned int width = original_data_tensor.width();
    unsigned int height = original_data_tensor.height();
    unsigned int original_width = original_data_tensor.width();
    unsigned int original_height = original_data_tensor.height();

    if (width & 1) width++; if (height & 1) height++;
    if (width & 2) width += 2; if (height & 2) height += 2;
    if (width & 4) width += 4; if (height & 4) height += 4;
    if (width & 8) width += 8; if (height & 8) height += 8;
    if (width & 16) width += 16; if (height & 16) height += 16;

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
        *helper_tensor.data_ptr(x, y, 0, 0) = ((Conv::datum) x) / ((Conv::datum) original_width - 1);
        *helper_tensor.data_ptr(x, y, 1, 0) = ((Conv::datum) y) / ((Conv::datum) original_height - 1);
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
    bool complete = factory->AddLayers(graph);
    if (!complete) FATAL("Failed completeness check, inspect model!");

    graph.Initialize();


    // Load network parameters
    graph.DeserializeParameters(param_tensor_file);

    graph.SetIsTesting(true);
    LOGINFO << "Classifying..." << std::flush;
    graph.FeedForward();

    Conv::Tensor *net_output_tensor = &graph.GetDefaultOutputNode()->output_buffers[0].combined_tensor->data; // &net.buffer(output_layer_id)->data;
    Conv::Tensor image_output_tensor(1, net_output_tensor->width(), net_output_tensor->height(), 3);

    LOGINFO << "Colorizing..." << std::flush;
    dataset->Colorize(*net_output_tensor, image_output_tensor);

    // Recrop image down
    Conv::Tensor small(1, original_data_tensor.width(), original_data_tensor.height(), 3);
    for (unsigned int m = 0; m < 3; m++)
      for (unsigned int y = 0; y < small.height(); y++)
        for (unsigned int x = 0; x < small.width(); x++)
          *small.data_ptr(x, y, m, 0) = *image_output_tensor.data_ptr_const(x, y, m, 0);

    if(argc > 5)
      small.WriteToFile(output_image_fname);
  } else if(dataset->GetTask() == Conv::CLASSIFICATION || dataset->GetTask() == Conv::DETECTION) {
    // Rescale image
    unsigned int width = dataset->GetWidth();
    unsigned int height = dataset->GetHeight();
    unsigned int original_width = original_data_tensor.width();
    unsigned int original_height = original_data_tensor.height();

    Conv::Tensor data_tensor(1, width, height, original_data_tensor.maps());
    data_tensor.Clear();
    Conv::Tensor::CopySample(original_data_tensor, 0, data_tensor, 0, false, true);

     // Assemble net
    Conv::NetGraph graph;
    Conv::InputLayer input_layer(data_tensor);

    Conv::NetGraphNode input_node(&input_layer);
    input_node.is_input = true;

    graph.AddNode(&input_node);
    bool complete = factory->AddLayers(graph);
    if (!complete) FATAL("Failed completeness check, inspect model!");

    graph.Initialize();

    // Load network parameters
    graph.DeserializeParameters(param_tensor_file);

    graph.SetIsTesting(true);
    LOGINFO << "Classifying..." << std::flush;
    graph.FeedForward();

    if(dataset->GetTask() == Conv::CLASSIFICATION) {
      Conv::Tensor *net_output_tensor = &graph.GetDefaultOutputNode()->output_buffers[0].combined_tensor->data;
      // TODO display scores
    } else {
      Conv::DatasetMetadataPointer* net_output = graph.GetDefaultOutputNode()->output_buffers[0].combined_tensor->metadata;
      std::vector<Conv::BoundingBox>* output_boxes = (std::vector<Conv::BoundingBox>*)net_output[0];

      LOGINFO << "Bounding boxes: " << output_boxes->size();
      for(unsigned int b = 0; b < output_boxes->size(); b++) {
        Conv::BoundingBox box = (*output_boxes)[b];
        box.x *= (Conv::datum)original_width;
        box.y *= (Conv::datum)original_height;
        box.w *= (Conv::datum)original_width;
        box.h *= (Conv::datum)original_height;

        LOGINFO << "Box\t" << b << ": " << class_manager.GetClassInfoById(box.c).first << " (" << box.score << ")";
        LOGINFO << "  Center: (" << box.x << "," << box.y << ")";
        LOGINFO << "  Size: (" << box.w << "x" << box.h << ")";

        // Draw box into original data tensor
        for(int bx = (box.x - (box.w / 2)); bx <= (box.x + (box.w / 2)); bx++) {
          int by_top = box.y - (box.h / 2);
          int by_bot = box.y + (box.h / 2);
          if(bx >= 0 && bx < original_width) {
            if (by_top >= 0 && by_top < original_height) {
              *(original_data_tensor.data_ptr(bx, by_top, 0, 0)) = 1.0;
              *(original_data_tensor.data_ptr(bx, by_top, 1, 0)) = 1.0;
              *(original_data_tensor.data_ptr(bx, by_top, 2, 0)) = 1.0;
            }
            if (by_bot >= 0 && by_bot < original_height) {
              *(original_data_tensor.data_ptr(bx, by_bot, 0, 0)) = 1.0;
              *(original_data_tensor.data_ptr(bx, by_bot, 1, 0)) = 1.0;
              *(original_data_tensor.data_ptr(bx, by_bot, 2, 0)) = 1.0;
            }
          }
        }
        // Draw vertical lines
        for(int by = (box.y - (box.h / 2)); by <= (box.y + (box.h / 2)); by++) {
          int bx_top = box.x - (box.w / 2);
          int bx_bot = box.x + (box.w / 2);
          if(by >= 0 && by < original_height) {
            if (bx_top >= 0 && bx_top < original_width) {
              *(original_data_tensor.data_ptr(bx_top, by, 0, 0)) = 1.0;
              *(original_data_tensor.data_ptr(bx_top, by, 1, 0)) = 1.0;
              *(original_data_tensor.data_ptr(bx_top, by, 2, 0)) = 1.0;
            }
            if (bx_bot >= 0 && bx_bot < original_width) {
              *(original_data_tensor.data_ptr(bx_bot, by, 0, 0)) = 1.0;
              *(original_data_tensor.data_ptr(bx_bot, by, 1, 0)) = 1.0;
              *(original_data_tensor.data_ptr(bx_bot, by, 2, 0)) = 1.0;
            }
          }
        }
      }

      if(argc > 5)
        original_data_tensor.WriteToFile(output_image_fname);
    }

  }

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
