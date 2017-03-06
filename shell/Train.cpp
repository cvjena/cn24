/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */

#include "ShellState.h"

namespace Conv {

CN24_SHELL_FUNC_IMPL(Train) {
  CN24_SHELL_FUNC_DESCRIPTION("Trains the network for a specified number of epochs");

  int epochs = 1;
  int do_snapshot = -1;
  int enable_training_stats = -1;

  cargo_add_option(cargo, (cargo_option_flags_t)0, "--detailed-statistics -d", "Enables calculation of all statistics "
    "instead of just training loss", "b", &enable_training_stats);

  cargo_add_option(cargo, (cargo_option_flags_t)0, "--do-snapshots -s", "Write metrics every epochs instead of after training only",
                   "b", &do_snapshot);

  cargo_add_option(cargo, (cargo_option_flags_t)CARGO_OPT_NOT_REQUIRED, "epochs", "Number of epochs to train the network for (default: 1)",
    "i", &epochs);
  cargo_add_validation(cargo, (cargo_validation_flags_t)0, "epochs", cargo_validate_int_range(1, 999999));

  CN24_SHELL_PARSE_ARGS;

  // Check if shell state allows for model loading
  if(state_ != NET_AND_TRAINER_LOADED) {
    LOGERROR << "Cannot train, no net is loaded or is loaded for prediction only.";
    return FAILURE;
  }

  // Check if there are any training samples
  unsigned int training_samples = 0;
  for(unsigned int i = 0; i < training_bundles_->size(); i++) {
    training_samples += training_bundles_->at(i)->GetSampleCount();
  }

  if(training_samples > 0) {
    trainer_->SetStatsDuringTraining(enable_training_stats == 1);
    trainer_->Train((unsigned int)epochs, do_snapshot == 1);
    return SUCCESS;
  } else {
    LOGWARN << "Training skipped, there were no training samples";
    return WRONG_PARAMS;
  }

}

}
