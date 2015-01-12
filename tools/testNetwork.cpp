/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file testNetwork.cpp
 * \brief Tests a convolutional neural net using a dataset tensor
 *
 * \author Clemens-A. Brust(ikosa.de@gmail.com)
 */

#include <iostream>
#include <fstream>
#include <vector>
#include <string>

#ifdef BUILD_OPENMP
#include <omp.h>
#endif

#include <cn24.h>

int main (int argc, char** argv) {
  // FIXME We're going to need a configuration framework at some point...
  const unsigned int CHANNELS = 3;
  const unsigned int LABEL_CHANNELS = 9;
  const unsigned int BATCHSIZE = 1024;
  const unsigned int GLOBAL_STEP = 13;
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

  if (argc < 4) {
    LOGERROR << "Usage: " << argv[0] <<
             " <net> <param tensor> <testing tensor> <experiment name> [<first image> <last image>]";
    LOGEND;
    return -1;
  }

  Conv::System::Init();

  // Load testing tensor
  std::ifstream testing_tensor (argv[3], std::ios::binary | std::ios::in);
  if (!testing_tensor.good()) {
    FATAL ("Cannot open " << argv[3] << "!");
    return -1;
  }

  // Load parameter tensor
  std::ifstream param_tensor (argv[2], std::ios::binary | std::ios::in);
  if (!param_tensor.good()) {
    FATAL ("Cannot open " << argv[2] << "!");
    return -1;
  }

  // Open log file
  std::ofstream output_file (argv[4], std::ios::out);

  Conv::Factory* factory = Conv::Factory::getNetFactory (argv[1][0], 49932);
  if (factory == nullptr) {
    FATAL ("Unknown net: " << argv[1]);
  }

  const unsigned int patchsize_x = factory->patchsizex();
  const unsigned int patchsize_y = factory->patchsizey();
  const unsigned int first_image = (argc < 7) ? 0 : atoi (argv[5]);
  const unsigned int last_image = (argc < 7) ? 32767 : atoi (argv[6]);

  Conv::Tensor dataA_tensor (BATCHSIZE, patchsize_x, patchsize_y, CHANNELS);
  Conv::Tensor dataB_tensor (BATCHSIZE, LABEL_CHANNELS);
  Conv::Tensor label_tensor (BATCHSIZE, 1, 1, 1);
  Conv::Tensor helper_tensor (BATCHSIZE, 2);
  Conv::Tensor weight_tensor (BATCHSIZE);

  Conv::Net netA;
  Conv::InputLayer inputA_layer (dataA_tensor, label_tensor, helper_tensor,
                                 weight_tensor);
  Conv::ConfusionMatrixLayer confusion_matrix_layer (class_names, LABEL_CHANNELS);

  int data_layer_id = netA.AddLayer (&inputA_layer);
  int output_layer_id =
    factory->AddLayers (netA, {Conv::Connection (data_layer_id) },
                        LABEL_CHANNELS);

  Conv::Net netB;
  Conv::InputLayer inputB_layer (dataB_tensor, label_tensor, helper_tensor, weight_tensor);
  int dataB_layer_id = netB.AddLayer (&inputB_layer);
  int confusion_layer_id =
  netB.AddLayer (&confusion_matrix_layer, {
    Conv::Connection (dataB_layer_id, 0),
    Conv::Connection (dataB_layer_id, 1),
    Conv::Connection (dataB_layer_id, 3)
  });

  // Load parameters
  netA.DeserializeParameters (param_tensor);
  Conv::Tensor* netA_output_tensor = & (netA.buffer (output_layer_id)->data);

  unsigned int current_image = 0;
  unsigned int images_done = 0;
  while (current_image <= last_image) {
    Conv::Tensor image_tensor;
    Conv::Tensor source_label_tensor;
    image_tensor.Deserialize (testing_tensor);
    source_label_tensor.Deserialize (testing_tensor);

    if (!testing_tensor.good())
      break;

    if (current_image >= first_image && current_image <= last_image) {
      LOGINFO << "Extracting image " << current_image << "..." << std::flush;

      Conv::Tensor extracted_data_tensor;
      Conv::Tensor extracted_helper_tensor;
      Conv::Tensor extracted_weight_tensor;
      Conv::Tensor extracted_label_tensor;

      // Extract data from images
      Conv::Segmentation::ExtractPatches (patchsize_x, patchsize_y,
                                          extracted_data_tensor,
                                          extracted_helper_tensor,
                                          image_tensor, 0, false);
      Conv::Segmentation::ExtractLabels (patchsize_x, patchsize_y,
                                         extracted_label_tensor,
                                         extracted_weight_tensor,
                                         source_label_tensor, 0, 0);

      unsigned int patches = extracted_data_tensor.samples();
      LOGINFO << "Feeding " << patches << " patches through the net...\n"
              << std::flush;

      Conv::Tensor saved_output_tensor (patches, LABEL_CHANNELS);
      Conv::Tensor saved_label_tensor (patches);
      Conv::Tensor saved_weight_tensor (patches);


      // First classification
      unsigned int fiftieth = 0;
      unsigned int tenth = 0;
      for (unsigned int sample = 0; sample < patches; sample += BATCHSIZE) {
#ifdef BUILD_OPENCL
        dataA_tensor.MoveToCPU (true);
        helper_tensor.MoveToCPU (true);
        weight_tensor.MoveToCPU (true);
        label_tensor.MoveToCPU (true);
#endif
        for (unsigned int s = 0; (s < BATCHSIZE) && ( (sample + s) < patches); s++) {
          Conv::Tensor::CopySample (extracted_data_tensor, sample + s, dataA_tensor, s);
          Conv::Tensor::CopySample (extracted_helper_tensor, sample + s, helper_tensor, s);
          Conv::Tensor::CopySample (extracted_weight_tensor, sample + s, weight_tensor, s);
          Conv::Tensor::CopySample (extracted_label_tensor, sample + s, label_tensor, s);
        }
        netA.FeedForward();

#ifdef BUILD_OPENCL
        netA_output_tensor->MoveToCPU();
#endif

        // Save output, weight and labels
        for (unsigned int s = 0; (s < BATCHSIZE) && ( (sample + s) < patches); s++) {
          Conv::Tensor::CopySample (*netA_output_tensor, s, saved_output_tensor, sample + s);
        }

        if ( ( (50 * sample) / patches) > fiftieth) {
          fiftieth = (50 * sample) / patches;
          std::cout << "." << std::flush;
        }
        if ( ( (10 * sample) / patches) > tenth) {
          tenth = (10 * sample) / patches;
          std::cout << tenth * 10 << "%" << std::flush;
        }
      }


      LOGINFO << "Processing...";
      // Processing
      // Conv::Segmentation::UseSLICO(patches, GLOBAL_STEP, LABEL_CHANNELS, saved_output_tensor, image_tensor);
      // Conv::Segmentation::UseFelzenszwalb(patches, 0.5, 550.0, 50, LABEL_CHANNELS, saved_output_tensor, image_tensor);

      LOGINFO << "Evaluating...\n";
      // Second classification
      fiftieth = 0;
      tenth = 0;

#ifdef BUILD_OPENCL
      dataB_tensor.MoveToCPU (true);
//      helper_tensor.MoveToCPU (true);
      weight_tensor.MoveToCPU (true);
      label_tensor.MoveToCPU (true);
#endif
      for (unsigned int sample = 0; sample < patches; sample += BATCHSIZE) {
	dataB_tensor.Clear(0);
	weight_tensor.Clear(0);
	label_tensor.Clear(0);
        for (unsigned int s = 0; (s < BATCHSIZE) && ( (sample + s) < patches); s++) {
          Conv::Tensor::CopySample (saved_output_tensor, sample + s, dataB_tensor, s);
//          Conv::Tensor::CopySample (extracted_helper_tensor, sample + s, helper_tensor, s);
          Conv::Tensor::CopySample (extracted_weight_tensor, sample + s, weight_tensor, s);
          Conv::Tensor::CopySample (extracted_label_tensor, sample + s, label_tensor, s);
        }
        netB.FeedForward();

        if ( ( (50 * sample) / patches) > fiftieth) {
          fiftieth = (50 * sample) / patches;
          std::cout << "." << std::flush;
        }
        if ( ( (10 * sample) / patches) > tenth) {
          tenth = (10 * sample) / patches;
          std::cout << tenth * 10 << "%" << std::flush;
        }
      }

      images_done++;

      if ( (images_done % 10) == 0) {
        confusion_matrix_layer.Print ("", false);
      }
    } else {
      LOGINFO << "Skipping image " << current_image;
    }

    current_image++;
  }

  confusion_matrix_layer.Print ("", false);
  confusion_matrix_layer.PrintCSV (output_file);


  output_file.close();
  param_tensor.close();
  testing_tensor.close();

  LOGEND;
}

