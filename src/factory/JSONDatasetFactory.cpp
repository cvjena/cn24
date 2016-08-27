/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Dataset.h"
#include "MNISTDataset.h"

#include "JSONDatasetFactory.h"

namespace Conv {
Dataset* JSONDatasetFactory::ConstructDataset(JSON descriptor) {
  if(descriptor.count("task") == 1) {
    std::string task = descriptor["task"];
    if(task.compare("segmentation") == 0) {
      JSONSegmentationDataset* segmentation_dataset = new JSONSegmentationDataset;
      segmentation_dataset->Load(descriptor, false, LOAD_BOTH);
      return segmentation_dataset;
    } else if(task.compare("classification") == 0) {

    } else if (task.compare("detection") == 0) {
      JSONDetectionDataset* detection_dataset = new JSONDetectionDataset;
      detection_dataset->Load(descriptor, false, LOAD_BOTH);
      return detection_dataset;
    } else {
      FATAL("Invalid task: " << task);
    }
  } else if(descriptor.count("special") == 1) {
    std::string special_dataset = descriptor["special"];
    if(special_dataset.compare("MNIST") == 0) {
      MNISTDataset* mnist_dataset = new MNISTDataset;
      mnist_dataset->Load(descriptor);
      return mnist_dataset;
    }
  } else {
    FATAL("Not a valid dataset (no task or special dataset)");
    return nullptr;
  }
}
}
