/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "MNISTDataset.h"

namespace Conv {

MNISTDataset::MNISTDataset() {
  for(unsigned int c = 0; c < 9; c++) {
    class_colors_.push_back(0);
    class_weights_.push_back(1.0);
    class_names_.push_back(std::to_string(c));
  }
}

void MNISTDataset::Load(JSON descriptor) {

}

bool MNISTDataset::GetTrainingSample(Tensor &data_tensor, Tensor &label_tensor, Tensor &helper_tensor,
                                     Tensor &weight_tensor, unsigned int sample, unsigned int index) {
  return false;
}

bool MNISTDataset::GetTestingSample(Tensor &data_tensor, Tensor &label_tensor, Tensor &helper_tensor,
                                    Tensor &weight_tensor, unsigned int sample, unsigned int index) {
  return false;
}
}
