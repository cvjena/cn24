/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */
#include <sstream>
#include <cmath>
#include <chrono>

#include "Log.h"
#include "NetGraph.h"
#include "NetGraphNode.h"
#include "StatLayer.h"
#include "LossFunctionLayer.h"
#include "CLHelper.h"
#include "StatAggregator.h"
#include "Init.h"
#include "JSONOptimizerFactory.h"

#include "Trainer.h"


namespace Conv {

bool Trainer::stats_are_initialized_ = false;
StatDescriptor* Trainer::stat_aggloss_ = nullptr;
StatDescriptor* Trainer::stat_sps_ = nullptr;
StatDescriptor* Trainer::stat_fps_ = nullptr;

template <typename T> int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

void Trainer::InitializeStats() {
  // Only initialize stats once
  if (!stats_are_initialized_) {

    stat_aggloss_ = new StatDescriptor;
    stat_aggloss_->nullable = true;
    stat_aggloss_->description = "Average Aggregate Loss";
    stat_aggloss_->unit = "1/pixel";
    stat_aggloss_->init_function =
      [](Stat& stat) {stat.is_null = true; stat.value = 0.0;};
    stat_aggloss_->update_function =
      [](Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
    stat_aggloss_->output_function =
      [](HardcodedStats& hc_stats, Stat& stat) -> Stat {
      Stat return_stat; return_stat.is_null = true;
      if (hc_stats.iterations > 0) {
        double d_iterations = (double)hc_stats.iterations;
        return_stat.value = stat.value / d_iterations;
        return_stat.is_null = false;
      }
      return return_stat;
    };

    stat_sps_ = new StatDescriptor;
    stat_sps_->nullable = true;
    stat_sps_->description = "Pixel Throughput";
    stat_sps_->unit = "pixels/s";
    stat_sps_->init_function =
      [](Stat& stat) {stat.is_null = true; stat.value = 0.0;};
    stat_sps_->update_function =
      [](Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
    stat_sps_->output_function =
      [] (Conv::HardcodedStats& hc_stats, Conv::Stat& stat) {
        Conv::Stat return_stat = stat;
        return_stat.value = stat.value / hc_stats.seconds_elapsed;
        return return_stat;
      };
    
    stat_fps_ = new StatDescriptor;
    stat_fps_->nullable = true;
    stat_fps_->description = "Frame Rate";
    stat_fps_->unit = "frames/s";
    stat_fps_->init_function =
      [](Stat& stat) {stat.is_null = true; stat.value = 0.0;};
    stat_fps_->update_function =
      [](Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
    stat_fps_->output_function =
      [] (Conv::HardcodedStats& hc_stats, Conv::Stat& stat) {
        Conv::Stat return_stat = stat;
        return_stat.value = stat.value / hc_stats.seconds_elapsed;
        return return_stat;
      };
    
    // Register stats
    System::stat_aggregator->RegisterStat(stat_aggloss_);
    System::stat_aggregator->RegisterStat(stat_sps_);
    System::stat_aggregator->RegisterStat(stat_fps_);
    stats_are_initialized_ = true;
  }
}

Trainer::Trainer(Conv::NetGraph& graph, JSON settings) :
  graph_(graph), settings_(settings) {
  LOGDEBUG << "Instance created";

  // We need a training layer to select training samples and some kind of
  // loss function to minimize
  if (graph_.GetTrainingNodes().size() == 0 || graph_.GetLossNodes().size() == 0) {
    FATAL("Net doesn't have training layer or loss function layer!");
  }

  // Ask the Net for parameters
  graph_.GetParameters(parameters_);
  LOGDEBUG << "Optimizing " << parameters_.size() << " sets of parameters.";

  // Set default optimizer to be SGDOptimizer
  if(!settings_.count("optimization_method")) settings_["optimization_method"] = "gd";

  // Create optimizer
  optimizer_ = JSONOptimizerFactory::ConstructOptimizer(settings_);
  if(optimizer_ == nullptr) {
    FATAL("Could not create optimizer!");
  }

  // Count weights and create accumulated gradient buffer
  unsigned int w = 0;
  for (unsigned int p = 0; p < parameters_.size(); p++) {
    w += parameters_[p]->data.elements();
    Tensor* accumulated_gradient = new Tensor();
    accumulated_gradient->Resize (parameters_[p]->data);
    accumulated_gradient->Clear();
    accumulated_gradients_.push_back (accumulated_gradient);
  }

  // Outputs the number of weights
  LOGDEBUG << "Weights: " << w;
  weight_count_ = w;

  first_training_layer_ = dynamic_cast<TrainingLayer*>(graph_.GetTrainingNodes()[0]->layer);
  sample_count_ = first_training_layer_->GetLabelWidth() * first_training_layer_->GetLabelHeight()
  * first_training_layer_->GetBatchSize();

  // Insert defaults
  if(!settings_.count("testing_ratio")) settings_["testing_ratio"] = 1.0;
  if(!settings_.count("epoch_training_ratio")) settings_["epoch_training_ratio"] = 1.0;
  if(!settings_.count("l1")) settings_["l1"] = 0.001;
  if(!settings_.count("l2")) settings_["l2"] = 0.0005;
  if(!settings_.count("batch_size_parallel")) settings_["batch_size_parallel"] = 1;
  if(!settings_.count("batch_size_sequential")) settings_["batch_size_sequential"] = 1;
  if(!settings_.count("epoch_iterations")) settings_["epoch_iterations"] = 500;
  if(!settings_.count("enable_stats_during_training")) settings_["enable_stats_during_training"] = true;

  InitializeStats();
}

void Trainer::UpdateParameterSizes() {
  unsigned int w = 0;

  for (unsigned int p = 0; p < parameters_.size(); p++) {
    w += parameters_[p]->data.elements();

    // Allocate Tensors for momentum
    Tensor* accumulated_gradient = accumulated_gradients_[p];

    if(accumulated_gradient->elements() != parameters_[p]->data.elements()) {
      accumulated_gradient->Resize(parameters_[p]->data);
      accumulated_gradient->Clear();
    }
  }

  if(w != weight_count_) {
    LOGDEBUG << "Weight count changed from " << weight_count_ << " to " << w;
    weight_count_ = w;
  }
}

void Trainer::Train (unsigned int epochs, bool do_snapshots) {
  // Update parameter sizes
  UpdateParameterSizes();

  // Update hardcoded stats
  System::stat_aggregator->hardcoded_stats_.weights = weight_count_;

  graph_.SetIsTesting(false);
  graph_.SetStatLayersEnabled(settings_["enable_stats_during_training"]);
  
  for (unsigned int e = 0; e < epochs; e++) {
    Epoch();
    if(do_snapshots) {
      System::stat_aggregator->Snapshot();
      // Update hardcoded stats
      System::stat_aggregator->hardcoded_stats_.weights = weight_count_;
    }
  }

  graph_.SetStatLayersEnabled(true);
}

void Trainer::Test() {
  // Update hardcoded stats
  System::stat_aggregator->hardcoded_stats_.weights = weight_count_;

	datum aggregate_loss = 0.0;
	datum* loss_sums = new datum[graph_.GetLossNodes().size()];
	for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++)
		loss_sums[n] = 0;

  unsigned int iterations = (first_training_layer_->GetSamplesInTestingSet()
                             / first_training_layer_->GetBatchSize()) + 1;
  iterations = (unsigned int) ( ( (datum) iterations) *
      (datum)settings_["testing_ratio"]);

	for (NetGraphNode* training_node : graph_.GetTrainingNodes())
		(dynamic_cast<TrainingLayer*>(training_node->layer))->SetTestingMode(true);

  graph_.SetIsTesting(true);

  LOGDEBUG << "Testing, iterations: " << iterations <<
           ", batch size: " << first_training_layer_->GetBatchSize();

  for (unsigned int i = 0; i < iterations; i++) {
    aggregate_loss = 0.0;

    first_training_layer_->SelectAndLoadSamples();
    graph_.FeedForward();

    for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
      LossFunctionLayer* lossfunction_layer = dynamic_cast<LossFunctionLayer*>(graph_.GetLossNodes()[n]->layer);
			const datum loss = lossfunction_layer->CalculateLossFunction();
			loss_sums[n] += loss;
			aggregate_loss += loss;
		}
    // Batch/Iteration done
    if (System::stat_aggregator->state_ == StatAggregator::RECORDING)
      System::stat_aggregator->hardcoded_stats_.iterations++;

    // Update aggregate loss stat
    System::stat_aggregator->Update(stat_aggloss_->stat_id, aggregate_loss
      / sample_count_ );

	}

  // Submit performance statistics
  System::stat_aggregator->Update(stat_sps_->stat_id, (double)sample_count_ * (double)iterations);
  System::stat_aggregator->Update(stat_fps_->stat_id, (double)(first_training_layer_->GetBatchSize()) * (double)iterations);

	for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
		LOGINFO << "Testing (Epoch " << epoch_ << ", node " << n << ") " << graph_.GetLossNodes()[n]->layer->GetLayerDescription() <<  " lps: " << loss_sums[n] / (datum)(iterations * sample_count_);
	}

	for (unsigned int n = 0; n < graph_.GetStatNodes().size(); n++) {
		StatLayer* stat_layer = dynamic_cast<StatLayer*>(graph_.GetStatNodes()[n]->layer);
    std::stringstream epochname;
    epochname << "Testing  - Epoch " << epoch_ << " -";
    stat_layer->UpdateAll();
    stat_layer->Print (epochname.str(), false);
	}

	for (NetGraphNode* training_node : graph_.GetTrainingNodes())
		(dynamic_cast<TrainingLayer*>(training_node->layer))->SetTestingMode(false);

	delete[] loss_sums;

  UpdateParameterSizes();
}

void Trainer::Epoch() {
  // Update hardcoded epoch stat
  System::stat_aggregator->hardcoded_stats_.epoch = epoch_;

	datum aggregate_loss = 0.0;
	datum* loss_sums = new datum[graph_.GetLossNodes().size()];
	for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++)
		loss_sums[n] = 0;

  unsigned int iterations =
      ((unsigned int)settings_["epoch_iterations"]) == (unsigned int)0 ?
      first_training_layer_->GetSamplesInTrainingSet() :
      (unsigned int)settings_["epoch_iterations"];
  iterations = (unsigned int) ( ( (datum) iterations) *
      (datum)settings_["epoch_training_ratio"]);

  unsigned int fiftieth = 0;
  unsigned int tenth = 0;

	for (NetGraphNode* training_node : graph_.GetTrainingNodes())
		(dynamic_cast<TrainingLayer*>(training_node->layer))->SetTestingMode(false);

  LOGINFO << "Epoch: " << epoch_ << ", it: " << iterations <<
           ", bsize: " << first_training_layer_->GetBatchSize() * (unsigned int)settings_["batch_size_sequential"]
          << ", " << optimizer_->GetStatusDescription(epoch_ * iterations) << std::endl << std::flush;

  for (unsigned int i = 0; i < iterations; i++) {
    if ( (50 * i / iterations) > fiftieth) {
      fiftieth = 50 * i / iterations;
      std::cout << "." << std::flush;
    }

    if ( (10 * i / iterations) > tenth) {
      tenth = 10 * i / iterations;
      std::cout << tenth << "0%" << std::flush;
    }
    aggregate_loss = 0.0;

    // Reset gradients
    for (unsigned int np = 0; np < accumulated_gradients_.size(); np++)
      accumulated_gradients_[np]->Clear();

    for (unsigned int b = 0; b < (unsigned int)settings_["batch_size_sequential"]; b++) {
      // Load data and feed forward
      first_training_layer_->SelectAndLoadSamples();
      graph_.FeedForward();
      UpdateParameterSizes();

      // Save errors
			for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
				LossFunctionLayer* lossfunction_layer = dynamic_cast<LossFunctionLayer*>(graph_.GetLossNodes()[n]->layer);
				const datum loss = lossfunction_layer->CalculateLossFunction();
				loss_sums[n] += loss;
				aggregate_loss += loss;
			}

      // Backpropagate errors
      graph_.BackPropagate();

      unsigned int np = 0;

      // Accumulate gradients
      for (unsigned int l = 0; l < graph_.GetNodes().size(); l++) {
				Layer* const layer = graph_.GetNodes()[l]->layer;
        for (unsigned int p = 0; p < layer->parameters().size(); p++) {
          Tensor& gradients = layer->parameters() [p]->delta;
#ifdef BUILD_OPENCL
          gradients.MoveToCPU();
#endif
          for (unsigned int e = 0; e < gradients.elements(); e++) {
            (* (accumulated_gradients_[np])) [e] += gradients[e];
          }
          np++;
        }
      }
    }
    // Apply regularization and local scaling
    ApplyRegularizationAndScaling();

    // Run the optimizer for a step
    optimizer_->Step(parameters_, epoch_ * iterations + i);

    // Batch/Iteration done
    if (System::stat_aggregator->state_ == StatAggregator::RECORDING)
      System::stat_aggregator->hardcoded_stats_.iterations++;

    // Update aggregate loss stat
    System::stat_aggregator->Update(stat_aggloss_->stat_id, aggregate_loss
      / (first_training_layer_->GetLossSamplingProbability() * sample_count_ * (datum)settings_["batch_size_sequential"]));

    // Notify update handler if possible
    if(update_handler != nullptr)
      update_handler->OnTrainerProgressUpdate((float)(i+1) / (float)iterations);
  }

  // Submit performance statistics
  System::stat_aggregator->Update(stat_sps_->stat_id, (double)sample_count_ * (double)iterations * (double)(settings_["batch_size_sequential"]));
  System::stat_aggregator->Update(stat_fps_->stat_id, (double)(first_training_layer_->GetBatchSize()) * (double)iterations * (double)(settings_["batch_size_sequential"]));
  
  // Display training epoch_error
	for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
		LOGINFO << "Training (Epoch " << epoch_ << ", node " << n << ") " << graph_.GetLossNodes()[n]->layer->GetLayerDescription() <<  " lps: " << loss_sums[n] / (datum)(iterations * sample_count_ * (datum)settings_["batch_size_sequential"] * first_training_layer_->GetLossSamplingProbability());
	}

  if(settings_["enable_stats_during_training"]) {
    for (unsigned int n = 0; n < graph_.GetStatNodes().size(); n++) {
      StatLayer* stat_layer = dynamic_cast<StatLayer*>(graph_.GetStatNodes()[n]->layer);
      std::stringstream epochname;
      epochname << "Training  - Epoch " << epoch_ << " -";
      stat_layer->UpdateAll();
      stat_layer->Print (epochname.str(), true);
    }
  }

  delete[] loss_sums;
  epoch_++;
}

void Trainer::ApplyRegularizationAndScaling() {
  unsigned int dp = 0;

  datum _cached_batch_size_sequential = settings_["batch_size_sequential"];
  datum _cached_l1_coefficient = settings_["l1"];
  datum _cached_l2_coefficient = settings_["l2"];

	for (unsigned int l = 0; l < graph_.GetNodes().size(); l++) {
		Layer* const layer = graph_.GetNodes()[l]->layer;
    datum local_learning_rate = layer->local_lr_;

    if(local_learning_rate == 0) {
      for (unsigned int p = 0; p < layer->parameters().size(); p++) {
        CombinedTensor *const current_layer_parameters = layer->parameters_[p];
        current_layer_parameters->delta.Clear();
        dp++;
      }
    } else {
      for (unsigned int p = 0; p < layer->parameters().size(); p++) {
        CombinedTensor *const current_layer_parameters = layer->parameters_[p];
  #ifdef BUILD_OPENCL
        current_layer_parameters->data.MoveToCPU();
        current_layer_parameters->delta.MoveToCPU(true);
  #endif

        for (unsigned int w = 0; w < current_layer_parameters->data.elements(); w++) {
          const datum weight = current_layer_parameters->data(w);

          // Gradients w.r.t. the weight
          const datum l1_gradient = (weight > 0) - (weight < 0);
          const datum l2_gradient = weight;
          const datum loss_gradient = (*accumulated_gradients_[dp])[w];

          const datum batch_size_loss_scaling_factor = ((datum) 1.0) /
                                                       (((datum) (sample_count_ * _cached_batch_size_sequential)) *
                                                        first_training_layer_->GetLossSamplingProbability());
          const datum partial_derivative =
              // Average of gradient over minibatch
              local_learning_rate * (loss_gradient * batch_size_loss_scaling_factor) +
              // Regularization
              local_learning_rate * (_cached_l2_coefficient * l2_gradient + _cached_l1_coefficient * l1_gradient);


          // Save partial derivative
          current_layer_parameters->delta[w] = partial_derivative;
        }
        dp++;
      }

    }
  }
}

std::ostream& operator<< (std::ostream & output,
                          const TrainerSettings settings) {
  output << "SB: " << settings.sbatchsize << ", ";
  output << "PB: " << settings.pbatchsize << ", ";
  output << "L1: " << settings.l1_weight << ", ";
  output << "L2: " << settings.l2_weight << ", ";
  return output;
}


}
