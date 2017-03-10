/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ShellState.h"

namespace Conv {

CN24_SHELL_FUNC_IMPL(Test) {
  CN24_SHELL_FUNC_DESCRIPTION("Evaluates the current network on bundles in the testing area");

  CN24_SHELL_PARSE_ARGS;

  // Check if shell state allows for model loading
  if(state_ != NET_AND_TRAINER_LOADED) {
    LOGERROR << "Cannot test, no net is loaded or is loaded for prediction only.";
    return FAILURE;
  }

  // Check if there are any testing samples
  unsigned int testing_samples = 0;
  for(unsigned int i = 0; i < testing_bundles_->size(); i++) {
    testing_samples += testing_bundles_->at(i)->GetSampleCount();
  }

  // TODO LayerView?

  if(testing_samples > 0) {
    // Remember old testing bundle
    int old_testing_bundle = input_layer_->GetActiveTestingSet();

    for(unsigned int d = 0; d < input_layer_->testing_sets_.size(); d++) {
      Bundle* bundle = input_layer_->testing_sets_[d];

      // Check if bundle has testing samples
      if(bundle->GetSegmentCount() > 0) {
        input_layer_->SetActiveTestingSet(d);
        // TODO Statistics recording on/off generate reset
        System::stat_aggregator->StartRecording();
        trainer_->Test();
        System::stat_aggregator->StopRecording();
        System::stat_aggregator->Generate();
        System::stat_aggregator->Reset();
      } else {
        LOGINFO << "Skipping bundle \"" << bundle->name << "\" because it has no samples";
      }
    }

    // Restore old testing bundle
    input_layer_->SetActiveTestingSet(old_testing_bundle);

    return SUCCESS;
  } else {
    LOGWARN << "Training skipped, there were no testing samples";
    return WRONG_PARAMS;
  }
}

}
