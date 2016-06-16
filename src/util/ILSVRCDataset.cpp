/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include <iostream>
#include <fstream>

#include "ILSVRCDataset.h"

namespace Conv {

ILSVRCDataset::ILSVRCDataset() {

}

void ILSVRCDataset::Load(JSON descriptor) {
  std::string path = descriptor["ilsvrc_path"];
  std::string descriptor_path = path + "/synsets.json";
  std::ifstream synset_file = std::ifstream(descriptor_path, std::ios::in);
  if(!synset_file.good()) {
    FATAL("Cannot open synset descriptor file, looking at " + path + "/synsets.json");
  }
  JSON synset_file_json = JSON::parse(synset_file);
  if(synset_file_json.count("synsets") == 0) {
    FATAL("No synsets found in descriptor file!");
  }
  JSON synsets_json = synset_file_json["synsets"];
  LOGINFO << "Loading " << synsets_json.size() << " synsets...";


  for(JSON::iterator synset_json_iterator = synsets_json.begin(); synset_json_iterator != synsets_json.end(); ++synset_json_iterator) {
    std::string WNID = synset_json_iterator.key();
    JSON synset_json = synset_json_iterator.value();
    LOGDEBUG << "Processing " << WNID << "...";
  }
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
