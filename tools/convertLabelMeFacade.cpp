/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file convertLabelMeFacade.cpp
 * \brief Program to convert the LabelMeFacade data to Tensor format.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

#include <cn24.h>

int main (int argc, char** argv) {
  if (argc < 4) {
    LOGINFO << "USAGE: " << argv[0] <<
            " <dataset location> <filelist> <name>";
    LOGEND;
    return -1;
  }

  std::string dataset_location (argv[1]);
  std::string filelist_name (argv[2]);
  std::string output_name (argv[3]);
  LOGDEBUG << "Dataset at " << dataset_location;
  LOGDEBUG << "Loading " << filelist_name;
  LOGDEBUG << "Writing to " << output_name;

  std::ifstream filelist (filelist_name, std::ios::in);
  std::ofstream output (output_name, std::ios::out | std::ios::binary);
  if (!filelist.good()) {
    FATAL ("Cannot open " << filelist_name);
  }
  if (!output.good()) {
    FATAL ("Cannot open " << output_name);
  }

  unsigned int images_done = 0;
  char line_buffer[16384];
  while (!filelist.eof()) {
    filelist.getline (line_buffer, 16384);
    if (filelist.fail()) {
      break;
    }

    std::stringstream image_name;
    std::stringstream label_name;
    image_name << dataset_location << "/images/" << line_buffer << ".jpg";
    label_name << dataset_location << "/labels/" << line_buffer << ".png";

    LOGINFO << "Reading " << image_name.str() << "...";
    Conv::Tensor image_tensor;
    Conv::JPGLoader::LoadFromFile (image_name.str(), image_tensor);
    image_tensor.Serialize (output);

    LOGINFO << "Reading " << label_name.str() << "...";
    Conv::Tensor label_tensor_rgb;
    std::ifstream label_file (label_name.str(), std::ios::in | std::ios::binary);
    Conv::PNGLoader::LoadFromStream (label_file, label_tensor_rgb);

    Conv::Tensor label_tensor (1, label_tensor_rgb.width(), label_tensor_rgb.height(), 1);
    for (unsigned int y = 0; y < label_tensor_rgb.height(); y++) {
      for (unsigned int x = 0; x < label_tensor_rgb.width(); x++) {
        /*
        [colors]
        0: various = 0:0:0
        1: building = 128:0:0
        2: car = 128:0:128
        3: door = 128:128:0
        4: pavement = 128:128:128
        5: road = 128:64:0
        6: sky = 0:128:128
        7: vegetation = 0:128:0
        8: window = 0:0:128
        */

        Conv::datum r = *label_tensor_rgb.data_ptr_const (x, y, 0);
        Conv::datum g = *label_tensor_rgb.data_ptr_const (x, y, 1);
        Conv::datum b = *label_tensor_rgb.data_ptr_const (x, y, 2);
        Conv::duint c = 0;

        {
          if (r < 0.25) {
            // r = 0
            if (g < 0.25) {
              // g = 0
              if (b < 0.25) {
                // b = 0
                c = 0;
              } else {
                // b = 128
                c = 8;
              }
            } else {
              // g = 128
              if (b < 0.25) {
                // b = 0
                c = 7;
              } else {
                // b = 128
                c = 6;
              }
            }
          } else {
            // r = 128
            if (g < 0.2) {
              // g = 0
              if (b < 0.25) {
                // b = 0
                c = 1;
              } else {
                // b = 128
                c = 2;
              }
            } else if (g < 0.4) {
              // g = 64
              if (b < 0.25) {
                // b = 0
                c = 5;
              } else {
                // b = 128
                FATAL ("Class does not exist");
              }
            } else {
              // g = 128
              if (b < 0.25) {
                // b = 0
                c = 3;
              } else {
                // b = 128
                c = 4;
              }
            }
          }

        }

        Conv::duint* target = (Conv::duint*) label_tensor.data_ptr(x, y);
        *target = c;
      }
    }

    label_tensor.Serialize (output);
    label_file.close();

    images_done++;
  }

  LOGINFO << "Converted " << images_done << " images";
  filelist.close();
  output.close();
  LOGEND;
  return 0;
}
