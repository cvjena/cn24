/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "Dataset.h"

#include "JSONDatasetFactory.h"

namespace Conv {
Dataset* JSONDatasetFactory::ConstructDataset(JSON descriptor) {
  if(descriptor.count("task") == 1) {
    std::string task = descriptor["task"];
    if(task.compare("segmentation") == 0) {
      JSONSegmentationDataset* segmentation_dataset = new JSONSegmentationDataset;
      segmentation_dataset->Load(descriptor, false, LOAD_BOTH);
      return segmentation_dataset;
    }
  } else {
    FATAL("Not a valid dataset (no task)");
    return nullptr;
  }
}
}
