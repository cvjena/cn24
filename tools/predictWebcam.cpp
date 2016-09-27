/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file predictWebcam.cpp
 * @brief Application that uses a pretrained net to show detections in webcam images
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <cstring>
#include <sstream>
#include <chrono>

#include <cn24.h>

#ifndef BUILD_OPENCV
int main (int argc, char* argv[]) {
  UNREFERENCED_PARAMETER(argc);
  UNREFERENCED_PARAMETER(argv);
  Conv::System::Init();
  LOGINFO << "Not built with OpenCV!";
  LOGEND;
}
#else
#include <opencv2/opencv.hpp>

int main (int argc, char* argv[]) {
  if (argc < 4) {
    LOGERROR << "USAGE: " << argv[0] << " <dataset config file> <net config file> <net parameter tensor>";
    LOGEND;
    return -1;
  }

  // Capture command line arguments
  std::string param_tensor_fname (argv[3]);
  std::string net_config_fname (argv[2]);
  std::string dataset_config_fname (argv[1]);
  
  // Initialize CN24
  Conv::System::Init();

  // Initialize Camera
  cv::VideoCapture capture;
  if(!capture.open(0)) {
    FATAL("Cannot open default camera device!");
    capture.set(CV_CAP_PROP_FRAME_WIDTH, 1280);
    capture.set(CV_CAP_PROP_FRAME_HEIGHT, 720);
    capture.set(CV_CAP_PROP_FPS, 30);
    capture.set(CV_CAP_PROP_CONVERT_RGB, true);
  }
  cv::namedWindow("Live stream", cv::WINDOW_AUTOSIZE);

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
  unsigned int CLASSES = class_manager.GetMaxClassId() + 1;

  if(dataset->GetTask() != Conv::DETECTION) {
    FATAL("Only detection is supported!");
  } else {
    // Rescale image
    unsigned int width = dataset->GetWidth();
    unsigned int height = dataset->GetHeight();

    // Load image
    cv::Mat first_camera_frame;
    capture >> first_camera_frame;
    LOGINFO << "First frame: " << first_camera_frame.cols << " x " << first_camera_frame.rows;
    Conv::Tensor original_data_tensor(1,first_camera_frame.cols,first_camera_frame.rows,3);

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
    bool complete = factory->AddLayers(graph, &class_manager);
    if (!complete) FATAL("Failed completeness check, inspect model!");

    graph.Initialize();


    // Load network parameters
    graph.DeserializeParameters(param_tensor_file);

    graph.SetIsTesting(true);
    LOGINFO << "Classifying..." << std::flush;

    auto start_time = std::chrono::system_clock::now();
    double avg_seconds = 0;

    while(true) {
      // Load cam image
      capture.grab();
      cv::Mat camera_frame;
      capture >> camera_frame;
#pragma omp parallel for default(shared)
      for(unsigned int y = 0; y < original_data_tensor.height(); y++) {
        const unsigned char* ptr = camera_frame.ptr<unsigned char>(y);
        for(unsigned int x = 0; x < original_data_tensor.width(); x++) {
          for(unsigned int c = 0; c < 3; c++) {
            unsigned int actual_channel = 2 - c;
            *(original_data_tensor.data_ptr(x, y, c)) = DATUM_FROM_UCHAR(ptr[x*3+actual_channel]);
          }
        }
      }
#ifdef BUILD_OPENCL
      data_tensor.MoveToCPU();
#endif
      Conv::Tensor::CopySample(original_data_tensor, 0, data_tensor, 0, false, true);

      // Predict
      graph.FeedForward();
      Conv::DatasetMetadataPointer* net_output = graph.GetDefaultOutputNode()->output_buffers[0].combined_tensor->metadata;
      std::vector<Conv::BoundingBox>* output_boxes = (std::vector<Conv::BoundingBox>*)net_output[0];

#pragma omp parallel for default(shared)
      for (unsigned int b = 0; b < output_boxes->size(); b++) {
        Conv::BoundingBox box = (*output_boxes)[b];
        box.x *= (Conv::datum) original_width;
        box.y *= (Conv::datum) original_height;
        box.w *= (Conv::datum) original_width;
        box.h *= (Conv::datum) original_height;

        // Draw box into original data tensor
        for (int bx = (box.x - (box.w / 2)); bx <= (box.x + (box.w / 2)); bx++) {
          int by_top = box.y - (box.h / 2);
          int by_bot = box.y + (box.h / 2);
          if (bx >= 0 && bx < original_width) {
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
        for (int by = (box.y - (box.h / 2)); by <= (box.y + (box.h / 2)); by++) {
          int bx_top = box.x - (box.w / 2);
          int bx_bot = box.x + (box.w / 2);
          if (by >= 0 && by < original_height) {
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

#pragma omp parallel for default(shared)
      for(unsigned int y = 0; y < original_data_tensor.height(); y++) {
        unsigned char* ptr = camera_frame.ptr<unsigned char>(y);
        for(unsigned int x = 0; x < original_data_tensor.width(); x++) {
          for(unsigned int c = 0; c < 3; c++) {
            unsigned int actual_channel = 2 - c;
            ptr[x*3+actual_channel] = UCHAR_FROM_DATUM(*(original_data_tensor.data_ptr_const(x, y, c)));
          }
        }
      }

      for (unsigned int b = 0; b < output_boxes->size(); b++) {
        Conv::BoundingBox box = (*output_boxes)[b];
        box.x *= (Conv::datum) original_width;
        box.y *= (Conv::datum) original_height;
        box.w *= (Conv::datum) original_width;
        box.h *= (Conv::datum) original_height;
        // Put box text
        cv::putText(camera_frame, class_manager.GetClassInfoById(box.c).first, cv::Point(box.x - (box.w/2), box.y + 20 - (box.h/2)), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255,255,255));
      }

      auto stop_time = std::chrono::system_clock::now();
      std::chrono::duration<double> t_diff = stop_time - start_time;
      start_time = stop_time;
      double seconds_elapsed = t_diff.count();
      avg_seconds = 0.9 * avg_seconds + 0.1 * seconds_elapsed;
      std::stringstream ss;
      ss << "FPS: " << std::setprecision(3) << 1.0 / avg_seconds;

      cv::putText(camera_frame, ss.str(), cv::Point(0, 20), cv::FONT_HERSHEY_PLAIN, 2.0, cv::Scalar(255,255,255));

      // Show image
      cv::imshow("Live stream", camera_frame);
      if(cv::waitKey(1) == 27)break;
    }
  }

  LOGINFO << "DONE!";
  LOGEND;
  return 0;
}
#endif
