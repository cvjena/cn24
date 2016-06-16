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
