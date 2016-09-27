/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <fstream>

#include "MNISTDataset.h"

namespace Conv {

MNISTDataset::MNISTDataset(ClassManager* class_manager) : Dataset(class_manager) {
  for(unsigned int c = 0; c <= 9; c++) {
    class_colors_.push_back(0);
    class_weights_.push_back(1.0);
    class_names_.push_back(std::to_string(c));
  }
}

void MNISTDataset::Load(JSON descriptor) {
  // TODO validate
  const unsigned int train_images_size = 47040016;
  const unsigned int train_labels_size = 60008;
  const unsigned int test_images_size = 7840016;
  const unsigned int test_labels_size = 10008;

  train_images_ = new uint8_t[train_images_size];
  train_labels_ = new uint8_t[train_labels_size];
  test_images_ = new uint8_t[test_images_size];
  test_labels_ = new uint8_t[test_labels_size];

  std::string path = descriptor["mnist_path"];
  std::ifstream train_images_stream(path + "/train-images-idx3-ubyte");
  std::ifstream train_labels_stream(path + "/train-labels-idx1-ubyte");
  std::ifstream test_images_stream(path + "/t10k-images-idx3-ubyte");
  std::ifstream test_labels_stream(path + "/t10k-labels-idx1-ubyte");

  if(!train_images_stream.good()) FATAL("Cannot open MNIST training images");
  if(!train_labels_stream.good()) FATAL("Cannot open MNIST training labels");
  if(!test_images_stream.good()) FATAL("Cannot open MNIST test images");
  if(!test_labels_stream.good()) FATAL("Cannot open MNIST test labels");

  train_images_stream.read((char*)train_images_, train_images_size);
  train_labels_stream.read((char*)train_labels_, train_labels_size);
  test_images_stream.read((char*)test_images_, test_images_size);
  test_labels_stream.read((char*)test_labels_, test_labels_size);

  if(*((uint32_t*)train_images_) != 0x03080000) FATAL("Wrong magic number in MNIST training images");
  if(*((uint32_t*)train_labels_) != 0x01080000) FATAL("Wrong magic number in MNIST training labels");
  if(*((uint32_t*)test_images_) != 0x03080000) FATAL("Wrong magic number in MNIST test images");
  if(*((uint32_t*)test_labels_) != 0x01080000) FATAL("Wrong magic number in MNIST test labels");
}

bool MNISTDataset::GetTrainingSample(Tensor &data_tensor, Tensor &label_tensor, Tensor &helper_tensor,
                                     Tensor &weight_tensor, unsigned int sample, unsigned int index) {
  UNREFERENCED_PARAMETER(helper_tensor);
  const std::size_t train_label_offset = 8;
  uint8_t label = train_labels_[train_label_offset + index];

  const std::size_t train_image_offset = 16;

  // Set image
  uint8_t* image = &(train_images_[train_image_offset + 28 * 28 * index]);
  for (unsigned int y = 0; y < 28; y++) {
    for (unsigned int x = 0; x < 28; x++) {
      const uint8_t image_byte = image[y * 28 + x];
      const datum image_datum = DATUM_FROM_UCHAR(image_byte);
      *(data_tensor.data_ptr(x, y, 0, sample)) = image_datum;
    }
  }

  // Set label
  label_tensor.Clear(0.0, sample);
  *(label_tensor.data_ptr(0,0,label,sample)) = 1.0;

  // Don't set helper

  // Set weight
  weight_tensor.Clear(1.0, sample);

  return true;
}

bool MNISTDataset::GetTestingSample(Tensor &data_tensor, Tensor &label_tensor, Tensor &helper_tensor,
                                    Tensor &weight_tensor, unsigned int sample, unsigned int index) {
  UNREFERENCED_PARAMETER(helper_tensor);
  const std::size_t test_label_offset = 8;
  uint8_t label = test_labels_[test_label_offset + index];

  const std::size_t test_image_offset = 16;

  // Set image
  uint8_t* image = &(test_images_[test_image_offset + 28 * 28 * index]);
  for (unsigned int y = 0; y < 28; y++) {
    for (unsigned int x = 0; x < 28; x++) {
      const uint8_t image_byte = image[y * 28 + x];
      const datum image_datum = DATUM_FROM_UCHAR(image_byte);
      *(data_tensor.data_ptr(x, y, 0, sample)) = image_datum;
    }
  }

  // Set label
  label_tensor.Clear(0.0, sample);
  *(label_tensor.data_ptr(0,0,label,sample)) = 1.0;

  // Don't set helper

  // Set weight
  weight_tensor.Clear(1.0, sample);

  return true;
}
}
