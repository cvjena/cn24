/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file Trainer.h
 * @class Trainer
 * @brief Trains a Net.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_TRAINER_H
#define CONV_TRAINER_H

#include <cmath>

#include "CombinedTensor.h"
#include "Net.h"

namespace Conv {

struct TrainerSettings {
public:
  datum learning_rate = 0.8;
  datum l1_weight = 0.001;
  datum l2_weight = 0.001;
  datum exponent = 0.75;
  datum gamma = 0.0001;
  datum momentum = 0.0;
  datum epoch_training_ratio = 1.0;
  datum testing_ratio = 1.0;
  unsigned int iterations = 0;
};

class Trainer {
public:
  /**
	* @brief Creates a Trainer for the specified Net
	*
	* @param net The Net to train
	* @param settings The settings to use in training
	*/
  Trainer (Net& net, TrainerSettings settings);

  /**
	* @brief Train the net for the specified number of epochs
	*
	* @param epochs The number of epochs to train
	*/
  void Train (unsigned int epochs);

  /**
	* @brief Test the net by running every test sample through the net
	*/
  datum Test();

  /**
	* @brief Train the net for exactly one epoch
	*/
  void Epoch();

  /**
	* @brief Set the current epoch
	*
	* @param epoch The new epoch number
	*/
  inline void SetEpoch (unsigned int epoch) {
    epoch_ = epoch;
  }

  inline datum CalculateLR (unsigned int iteration) {
    return settings_.learning_rate * pow (1.0 + settings_.gamma
                                          * (datum) iteration,
                                          -settings_.exponent);
  }

private:
  void ApplyGradients (datum lr);
  // References for easy access
  Net& net_;
  std::vector<CombinedTensor*> parameters_;
  std::vector<Tensor*> last_deltas_;
  TrainingLayer* training_layer_;
  LossFunctionLayer* lossfunction_layer_;

  // Learning options
  TrainerSettings settings_;

  // State
  unsigned int epoch_ = 0;
};


/**
 * @brief Prints settings to the ostream, may be helpful.
 */
std::ostream& operator<< (std::ostream& output, const TrainerSettings settings);

}

#endif
