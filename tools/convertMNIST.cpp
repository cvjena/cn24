/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file convertMNIST.cpp
 * \brief Program to convert MNIST data to Tensor format.
 * 
 * You can find the dataset at http://yann.lecun.com/exdb/mnist/
 * 
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#include <iostream>
#include <algorithm>
#include <fstream>
#include <cstdint>

#include <cn24.h>

// FIXME remove this function because it is stole from SO.
template <class T>
void endswap(T *objp)
{
  unsigned char *memp = reinterpret_cast<unsigned char*>(objp);
  std::reverse(memp, memp + sizeof(T));
}

int main(int argc, char* argv[]) {
  if(argc != 4) {
    LOGERROR << "USAGE: " << argv[0] << "<data> <labels> <output>";
    LOGEND;
    return -1;
  }
  
  /*
   * Convert data
   */
  std::ifstream data_file(argv[1], std::ios::in | std::ios::binary);
  data_file.ignore(4);
  
  uint32_t images = 0;
  uint32_t rows = 0;
  uint32_t columns = 0;
  bool little = false;
  data_file.read((char*)&images, sizeof(uint32_t)/sizeof(char));
  
  if(images != 60000 && images != 10000) {
    LOGINFO << "Little endian.";
    little = true;
    endswap(&images);
  }
  
  data_file.read((char*)&rows, sizeof(uint32_t)/sizeof(char));
  data_file.read((char*)&columns, sizeof(uint32_t)/sizeof(char));
  
  if(little) {
    endswap(&rows);
    endswap(&columns);
  }
  
  LOGINFO << "Converting " << images << " images...";
  
  Conv::Tensor data(images, columns + 4, rows + 4, 1);
  data.Clear();
  
  // This part adds a border around the image so that the end result is
  // a 32x32 image. It improves the performance of the LeNet-5 architecture.
  unsigned char d = 0;
  for(std::size_t i = 0; i < images; i++) {
    for(std::size_t y = 0; y < rows; y++) {
      for(std::size_t x = 0; x < columns; x++) {
        data_file.read((char*)&d, 1);
        *data.data_ptr(x + 2, y + 2, 0, i) = DATUM_FROM_UCHAR(d);
      }
    }
  }
  
  data_file.close();
  
  /*
   * Convert labels
   */
  std::ifstream label_file(argv[2], std::ios::in | std::ios::binary);
  label_file.ignore(4);
  
  uint32_t items = 0;
  label_file.read((char*)&items, sizeof(uint32_t) / sizeof(char));
  
  if(little)
    endswap(&items);
  
  LOGINFO << "Converting " << items << " labels...";
  
  if(items != images)
    FATAL("Label dataset size doesn't match image dataset!");
  
  Conv::Tensor labels(items, 10, 1, 1);
  labels.Clear(0);
  
  for(std::size_t e = 0; e < labels.samples(); e++) {
    label_file.read((char*)&d, 1);
    *labels.data_ptr((std::size_t)d, 0, 0, e) = 1.0;
  }
  
  label_file.close();
  
  LOGINFO << "Serializing to " << argv[3];
  
  std::ofstream output(argv[3], std::ios::out | std::ios::binary);
  
  data.Serialize(output);
  labels.Serialize(output);
  
  output.close();
  
  LOGEND;
  
}