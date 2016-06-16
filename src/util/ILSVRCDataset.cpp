/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <fstream>

#include "ILSVRCDataset.h"

namespace Conv {

ILSVRCDataset::ILSVRCDataset() {

}

void ILSVRCDataset::Load(JSON descriptor) {
  std::string path = descriptor["ilsvrc_path"];
  std::ifstream synset_file = std::ifstream(path + "/synsets.json", std::ios::in);
  if(!synset_file.good()) {
    FATAL("Cannot open synset descriptor file, looking at " + path + "/synsets.json");
  }
  JSON synset_json = JSON::parse(synset_file);
  if(synset_json.count("synsets") == 0) {
    FATAL("No synsets found in descriptor file!");
  }
  JSON synsets = synset_json["synsets"];
  LOGINFO << "Loading " << synsets.size() << " synsets...";
  classes_ = 1;
  testing_samples_ = 0;
  training_samples_ = 0;
}

bool ILSVRCDataset::GetTrainingSample(Tensor &data_tensor, Tensor &label_tensor, Tensor &helper_tensor,
                                     Tensor &weight_tensor, unsigned int sample, unsigned int index) {
  return false;
}

bool ILSVRCDataset::GetTestingSample(Tensor &data_tensor, Tensor &label_tensor, Tensor &helper_tensor,
                                    Tensor &weight_tensor, unsigned int sample, unsigned int index) {
  return false;
}
}
