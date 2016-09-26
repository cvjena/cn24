/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "YOLODynamicOutputLayer.h"

namespace Conv {

YOLODynamicOutputLayer::YOLODynamicOutputLayer(JSON configuration, ClassManager *class_manager) :
  SimpleLayer(configuration), class_manager_(class_manager) {

}

bool YOLODynamicOutputLayer::Connect(const CombinedTensor *input, CombinedTensor *output) {
  return true;
}

bool YOLODynamicOutputLayer::CreateOutputs(const std::vector<CombinedTensor *> &inputs,
                                           std::vector<CombinedTensor *> &outputs) {
  return false;
}

void YOLODynamicOutputLayer::FeedForward() {

}

void YOLODynamicOutputLayer::BackPropagate() {

}
}