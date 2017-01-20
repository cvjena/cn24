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
#include "../util/JSONParsing.h"

#include "../util/CombinedTensor.h"
#include "../util/StatAggregator.h"
#include "../math/Optimizer.h"
#include "TrainingLayer.h"
#include "NetGraph.h"

namespace Conv {

enum OPTIMIZATION_METHOD {
  GRADIENT_DESCENT,
  QUICKPROP
};
  
struct TrainerSettings {
public:
  datum l1_weight = 0.001;
  datum l2_weight = 0.0005;
  datum epoch_training_ratio = 1.0;
  datum testing_ratio = 1.0;
  bool stats_during_training = true;
  unsigned int pbatchsize = 1;
  unsigned int sbatchsize = 1;
  unsigned int iterations = 500;
};

class TrainerProgressUpdateHandler {
public:
  virtual void OnTrainerProgressUpdate(datum progress) = 0;
};

class Trainer {
public:
  /**
	* @brief Creates a Trainer for the specified Net
	*
	* @param net The Net to train
	* @param settings The settings to use in training
	*/
  Trainer (NetGraph& graph, JSON settings);

  ~Trainer() {
    delete optimizer_;
  }

  /**
	* @brief Train the net for the specified number of epochs
	*
	* @param epochs The number of epochs to train
	*/
  void Train (unsigned int epochs, bool do_snapshot);

  /**
	* @brief Test the net by running every test sample through the net
	*/
  void Test();

  /**
	* @brief Train the net for exactly one epoch
	*/
  void Epoch();

  /**
    @brief Resets the Trainer's inner state (last gradients/steps)
  **/
  inline void Reset() {
    LOGDEBUG << "Resetting Trainer and Optimizer state";
    optimizer_->Reset();
    first_iteration = true;
  }

  /**
	* @brief Set the current epoch
	*
	* @param epoch The new epoch number
	*/
  inline void SetEpoch (unsigned int epoch) {
    if(epoch != epoch_) {
      epoch_ = epoch;
      Reset();
    }
  }
  
  inline unsigned int epoch() { return epoch_; }

  inline void SetStatsDuringTraining(bool enable) { settings_["enable_stats_during_training"] = enable; }

  void UpdateParameterSizes();

  inline void SetUpdateHandler(TrainerProgressUpdateHandler* update_handler) { this->update_handler = update_handler; }

  JSON& settings() { return settings_; }
private:
  void ApplyRegularizationAndScaling();
  void InitializeStats();

  // References for easy access
  NetGraph& graph_;
  std::vector<CombinedTensor*> parameters_;
  std::vector<Tensor*> accumulated_gradients_;

  // Optimizer
  Optimizer* optimizer_;
  
	// Saved pointers
	TrainingLayer* first_training_layer_ = nullptr;

  // Sample count
  unsigned int sample_count_ = 0;
  unsigned int weight_count_ = 0;

  // Learning options
  JSON settings_;

  // State
  unsigned int epoch_ = 0;
  bool first_iteration = true;

  // Update handler
  TrainerProgressUpdateHandler* update_handler = nullptr;

  // Global state
  static bool stats_are_initialized_;
  static StatDescriptor* stat_aggloss_;
  static StatDescriptor* stat_fps_;
  static StatDescriptor* stat_sps_;
};


/**
 * @brief Prints settings to the ostream, may be helpful.
 */
std::ostream& operator<< (std::ostream& output, const TrainerSettings settings);

}

#endif
