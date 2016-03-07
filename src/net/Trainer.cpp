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

#include "Trainer.h"


namespace Conv {

bool Trainer::stats_are_initialized_ = false;
StatDescriptor* Trainer::stat_aggloss_ = nullptr;
StatDescriptor* Trainer::stat_qp_caseA_ = nullptr;
StatDescriptor* Trainer::stat_qp_caseB_ = nullptr;
StatDescriptor* Trainer::stat_qp_caseC_ = nullptr;
StatDescriptor* Trainer::stat_qp_caseM_ = nullptr;
StatDescriptor* Trainer::stat_sps_ = nullptr;
StatDescriptor* Trainer::stat_fps_ = nullptr;

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
      
    stat_qp_caseA_ = new StatDescriptor;
    stat_qp_caseA_->nullable = true;
    stat_qp_caseA_->description = "QuickProp Case A Percentage";
    stat_qp_caseA_->unit = "%";
    stat_qp_caseA_->init_function =
      [](Stat& stat) {stat.is_null = true; stat.value = 0.0;};
    stat_qp_caseA_->update_function =
      [](Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
    stat_qp_caseA_->output_function = 
      [](HardcodedStats& hc_stats, Stat& stat) -> Stat {
      Stat return_stat; return_stat.is_null = true;
      if (hc_stats.iterations > 0 && hc_stats.weights > 0 && !stat.is_null) {
        double d_iterations = (double)hc_stats.iterations;
        double d_weights = (double)hc_stats.weights;
        return_stat.value = 100.0 * stat.value / (d_iterations * d_weights);
        return_stat.is_null = false;
      }
      return return_stat;
    };
    
    stat_qp_caseB_ = new StatDescriptor;
    stat_qp_caseB_->nullable = true;
    stat_qp_caseB_->description = "QuickProp Case B Percentage";
    stat_qp_caseB_->unit = "%";
    stat_qp_caseB_->init_function =
      [](Stat& stat) {stat.is_null = true; stat.value = 0.0;};
    stat_qp_caseB_->update_function =
      [](Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
    stat_qp_caseB_->output_function = 
      [](HardcodedStats& hc_stats, Stat& stat) -> Stat {
      Stat return_stat; return_stat.is_null = true;
      if (hc_stats.iterations > 0 && hc_stats.weights > 0 && !stat.is_null) {
        double d_iterations = (double)hc_stats.iterations;
        double d_weights = (double)hc_stats.weights;
        return_stat.value = 100.0 * stat.value / (d_iterations * d_weights);
        return_stat.is_null = false;
      }
      return return_stat;
    };
    
    stat_qp_caseC_ = new StatDescriptor;
    stat_qp_caseC_->nullable = true;
    stat_qp_caseC_->description = "QuickProp Case C Percentage";
    stat_qp_caseC_->unit = "%";
    stat_qp_caseC_->init_function =
      [](Stat& stat) {stat.is_null = true; stat.value = 0.0;};
    stat_qp_caseC_->update_function =
      [](Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
    stat_qp_caseC_->output_function = 
      [](HardcodedStats& hc_stats, Stat& stat) -> Stat {
      Stat return_stat; return_stat.is_null = true;
      if (hc_stats.iterations > 0 && hc_stats.weights > 0 && !stat.is_null) {
        double d_iterations = (double)hc_stats.iterations;
        double d_weights = (double)hc_stats.weights;
        return_stat.value = 100.0 * stat.value / (d_iterations * d_weights);
        return_stat.is_null = false;
      }
      return return_stat;
    };
    
    stat_qp_caseM_ = new StatDescriptor;
    stat_qp_caseM_->nullable = true;
    stat_qp_caseM_->description = "QuickProp Case M Percentage";
    stat_qp_caseM_->unit = "%";
    stat_qp_caseM_->init_function =
      [](Stat& stat) {stat.is_null = true; stat.value = 0.0;};
    stat_qp_caseM_->update_function =
      [](Stat& stat, double user_value) {stat.value += user_value; stat.is_null = false;};
    stat_qp_caseM_->output_function = 
      [](HardcodedStats& hc_stats, Stat& stat) -> Stat {
      Stat return_stat; return_stat.is_null = true;
      if (hc_stats.iterations > 0 && hc_stats.weights > 0 && !stat.is_null) {
        double d_iterations = (double)hc_stats.iterations;
        double d_weights = (double)hc_stats.weights;
        return_stat.value = 100.0 * stat.value / (d_iterations * d_weights);
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
    System::stat_aggregator->RegisterStat(stat_qp_caseA_);
    System::stat_aggregator->RegisterStat(stat_qp_caseB_);
    System::stat_aggregator->RegisterStat(stat_qp_caseC_);
    System::stat_aggregator->RegisterStat(stat_qp_caseM_);
    System::stat_aggregator->RegisterStat(stat_sps_);
    System::stat_aggregator->RegisterStat(stat_fps_);
    stats_are_initialized_ = true;
  }
  
  // Move lambdas with reference captures here
}

Trainer::Trainer(Conv::NetGraph& graph, TrainerSettings settings) :
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

  unsigned int w = 0;

  for (unsigned int p = 0; p < parameters_.size(); p++) {
    w += parameters_[p]->data.elements();

    // Allocate Tensors for momentum
    Tensor* last_delta = new Tensor();
    Tensor* last_gradient = new Tensor();
    Tensor* accumulated_gradient = new Tensor();
    last_delta->Resize (parameters_[p]->data);
    last_delta->Clear();
    last_gradient->Resize (parameters_[p]->data);
    last_gradient->Clear();
    accumulated_gradient->Resize (parameters_[p]->data);
    accumulated_gradient->Clear();

    last_deltas_.push_back (last_delta);
    last_gradients_.push_back (last_gradient);
    accumulated_gradients_.push_back (accumulated_gradient);
  }

  // Outputs the number of weights
  LOGDEBUG << "Weights: " << w;
  weight_count_ = w;

  first_training_layer_ = dynamic_cast<TrainingLayer*>(graph_.GetTrainingNodes()[0]->layer);
  sample_count_ = first_training_layer_->GetLabelWidth() * first_training_layer_->GetLabelHeight()
  * first_training_layer_->GetBatchSize();

  InitializeStats();
}

void Trainer::Train (unsigned int epochs, bool do_snapshots) {
  // Update hardcoded stats
  System::stat_aggregator->hardcoded_stats_.weights = weight_count_;

  graph_.SetIsTesting(false);
  graph_.SetStatLayersEnabled(settings_.stats_during_training);
  
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
                                settings_.testing_ratio);

	for (NetGraphNode* training_node : graph_.GetTrainingNodes())
		(dynamic_cast<TrainingLayer*>(training_node->layer))->SetTestingMode(true);

  graph_.SetIsTesting(true);

  LOGDEBUG << "Testing, iterations: " << iterations <<
           ", batch size: " << first_training_layer_->GetBatchSize();

  for (unsigned int i = 0; i < iterations; i++) {
    aggregate_loss = 0.0;
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
}

void Trainer::Epoch() {
  // Update hardcoded epoch stat
  System::stat_aggregator->hardcoded_stats_.epoch = epoch_;

	datum aggregate_loss = 0.0;
	datum* loss_sums = new datum[graph_.GetLossNodes().size()];
	for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++)
		loss_sums[n] = 0;

  unsigned int iterations =
    settings_.iterations == 0 ?
    first_training_layer_->GetSamplesInTrainingSet() :
    settings_.iterations;
  iterations = (unsigned int) ( ( (datum) iterations) *
                                settings_.epoch_training_ratio);

  unsigned int fiftieth = 0;
  unsigned int tenth = 0;

	for (NetGraphNode* training_node : graph_.GetTrainingNodes())
		(dynamic_cast<TrainingLayer*>(training_node->layer))->SetTestingMode(false);

  LOGINFO << "Epoch: " << epoch_ << ", it: " << iterations <<
           ", bsize: " << first_training_layer_->GetBatchSize() * settings_.sbatchsize << ", current lr: " <<
           CalculateLR (epoch_ * iterations) << std::endl;

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

    for (unsigned int b = 0; b < settings_.sbatchsize; b++) {
      graph_.FeedForward();

      // Save errors
			for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
				LossFunctionLayer* lossfunction_layer = dynamic_cast<LossFunctionLayer*>(graph_.GetLossNodes()[n]->layer);
				const datum loss = lossfunction_layer->CalculateLossFunction();
				loss_sums[n] += loss;
				aggregate_loss += loss;
			}

      // Correct errors
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
    // Calculate annealed learning rate
    const datum lr =
      CalculateLR (epoch_ * iterations + i);

    // Apply gradients with new learning rate
    ApplyGradients (lr);

    // Batch/Iteration done
    if (System::stat_aggregator->state_ == StatAggregator::RECORDING)
      System::stat_aggregator->hardcoded_stats_.iterations++;

    // Update aggregate loss stat
    System::stat_aggregator->Update(stat_aggloss_->stat_id, aggregate_loss
      / (first_training_layer_->GetLossSamplingProbability() * sample_count_ * settings_.sbatchsize));
  }

  // Submit performance statistics
  System::stat_aggregator->Update(stat_sps_->stat_id, (double)sample_count_ * (double)iterations * (double)(settings_.sbatchsize));
  System::stat_aggregator->Update(stat_fps_->stat_id, (double)(first_training_layer_->GetBatchSize()) * (double)iterations * (double)(settings_.sbatchsize));
  
  // Display training epoch_error
	for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
		LOGINFO << "Training (Epoch " << epoch_ << ", node " << n << ") " << graph_.GetLossNodes()[n]->layer->GetLayerDescription() <<  " lps: " << loss_sums[n] / (datum)(iterations * sample_count_ * settings_.sbatchsize * first_training_layer_->GetLossSamplingProbability());
	}

  if(settings_.stats_during_training) {
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

void Trainer::ApplyGradients (datum lr) {
  unsigned int dp = 0;
  unsigned int qp_caseA = 0, qp_caseB = 0, qp_caseC = 0, qp_caseM = 0;
  
	for (unsigned int l = 0; l < graph_.GetNodes().size(); l++) {
		Layer* const layer = graph_.GetNodes()[l]->layer;
    datum layer_lr;
    switch (settings_.optimization_method) {
      case GRADIENT_DESCENT:
        layer_lr = layer->local_lr_;
        break;
      case QUICKPROP:
        layer_lr = layer->local_lr_;
        break;
    }

    for (unsigned int p = 0; p < layer->parameters().size(); p++) {
      CombinedTensor* const param = layer->parameters_[p];
#ifdef BUILD_OPENCL
      param->data.MoveToCPU();
#endif

      for (unsigned int w = 0; w < param->data.elements(); w++) {
        const datum weight = param->data (w);
        const datum l1_gradient = (weight > 0) - (weight < 0);
        const datum l2_gradient = weight;
        const datum w_gradient = (*accumulated_gradients_[dp]) [w];

        /*
         * http://www.iro.umontreal.ca/~pift6266/H10/notes/gradient.html
         *
         * This site says that one should average the gradient over
         * the minibatch
         */
        datum delta =
        
          // Average of gradient over minibatch
          layer_lr * (w_gradient / ((datum) (sample_count_ * settings_.sbatchsize)) * first_training_layer_->GetLossSamplingProbability()) +
          // Regularization
          layer_lr * (settings_.l2_weight * l2_gradient + settings_.l1_weight * l1_gradient);
        
        // This is needed for both methods
        const datum last_step = (*last_deltas_[dp]) (w);
        
        switch (settings_.optimization_method) {
          case GRADIENT_DESCENT:
          {
            const datum step = lr * delta + settings_.momentum * last_step;
            param->data[w] -= step;

            // Backup delta
            (*last_deltas_[dp]) [w] = step;
          }
            break;
          case QUICKPROP:
          {
            const datum last_gradient = (*last_gradients_[dp]) (w);
            const datum s = settings_.mu / (1.0 + settings_.mu);
            
            datum step = 0;
            if(last_step > 0.00001) {
              if(delta > 0.0) {
                step += lr * settings_.eta * delta;
                qp_caseB++;
              }
              
              if(delta > (s * last_gradient)) {
                step += settings_.mu * last_step;
                qp_caseM++;
              } else {
                step += last_step * delta / (last_gradient - delta);
              }
              qp_caseA++;
              
            } else if(last_step < -0.00001) {
              if(delta < 0.0) {
                step += lr * settings_.eta * delta;
                qp_caseB++;
              }
              
              if(delta < (s * last_gradient)) {
                step += settings_.mu * last_step;
                qp_caseM++;
              } else {
                step += last_step * delta / (last_gradient - delta);
              }
              qp_caseA++;
            } else {
              step += lr * settings_.eta * delta;
              qp_caseC++;
            }
            
            if(step > 1000 || step < -1000) {
              if(step>1000)
                step=1000;
              else
                step=-1000;
            }
            
            param->data[w] -= step;

            // Backup steps and gradient
            (*last_deltas_[dp]) [w] = step;
            (*last_gradients_[dp]) [w] = delta;
          }
            break;
        }
      }

      dp++;
    }
  }
  
  // Update quickprop stats
  if(settings_.optimization_method == QUICKPROP) {
    System::stat_aggregator->Update(stat_qp_caseA_->stat_id, (double)qp_caseA);
    System::stat_aggregator->Update(stat_qp_caseB_->stat_id, (double)qp_caseB);
    System::stat_aggregator->Update(stat_qp_caseC_->stat_id, (double)qp_caseC);
    System::stat_aggregator->Update(stat_qp_caseM_->stat_id, (double)qp_caseM);
  }
}

std::ostream& operator<< (std::ostream & output,
                          const TrainerSettings settings) {
  output << "LR: " << settings.learning_rate << ", ";
  output << "GM: " << settings.gamma << ", ";
  output << "EX: " << settings.exponent << ", ";
  output << "SB: " << settings.sbatchsize << ", ";
  output << "PB: " << settings.pbatchsize << ", ";
  output << "L1: " << settings.l1_weight << ", ";
  output << "L2: " << settings.l2_weight << ", ";
  output << "MM: " << settings.momentum << ", ";
  output << "MU: " << settings.mu << ", ";
  output << "ET: " << settings.eta << ", ";
  switch (settings.optimization_method) {
    case GRADIENT_DESCENT:
      output << "GD";
      break;
    case QUICKPROP:
      output << "QP";
      break;
  }
  return output;
}


}
