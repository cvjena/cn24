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
#include "Net.h"
#include "StatLayer.h"
#include "CLHelper.h"

#include "Trainer.h"

namespace Conv {

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

		first_training_layer_ = dynamic_cast<TrainingLayer*>(graph_.GetTrainingNodes()[0]->layer);
		sample_count_ = first_training_layer_->GetLabelWidth() * first_training_layer_->GetLabelHeight()
    * first_training_layer_->GetBatchSize();
}

void Trainer::Train (unsigned int epochs) {
  // net_.SetTestOnlyStatDisabled (false);
  graph_.SetIsTesting(false);

  for (unsigned int e = 0; e < epochs; e++)
    Epoch();

  // net_.SetTestOnlyStatDisabled (false);
}

void Trainer::Test() {
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

  auto t_begin = std::chrono::system_clock::now();

  for (unsigned int i = 0; i < iterations; i++) {
    graph_.FeedForward();
    for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
      LossFunctionLayer* lossfunction_layer = dynamic_cast<LossFunctionLayer*>(graph_.GetLossNodes()[n]->layer);
			const datum loss = lossfunction_layer->CalculateLossFunction();
			loss_sums[n] += loss;
			aggregate_loss += loss;
		}
	}

  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> t_diff = t_end - t_begin;
  LOGDEBUG << "Testing, sps: " <<
          (datum) (sample_count_ * iterations)
          / (datum) t_diff.count();

  LOGDEBUG << "Testing, tps: " <<
          1000000.0f * (datum) t_diff.count() /
          (datum) (sample_count_ * iterations) << " us";

	for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
		LossFunctionLayer* lossfunction_layer = dynamic_cast<LossFunctionLayer*>(graph_.GetLossNodes()[n]->layer);
		LOGINFO << "Testing (Epoch " << epoch_ << ", node " << n << ") " << graph_.GetLossNodes()[n]->layer->GetLayerDescription() <<  " lps: " << loss_sums[n] / (datum)(iterations * sample_count_);
	}
	LOGINFO << "Testing (Epoch " << epoch_ << ") aggregate lps: " << aggregate_loss / (datum)(iterations * sample_count_);

	for (unsigned int n = 0; n < graph_.GetStatNodes().size(); n++) {
		StatLayer* stat_layer = dynamic_cast<StatLayer*>(graph_.GetStatNodes()[n]->layer);
    std::stringstream epochname;
    epochname << "Testing  - Epoch " << epoch_ << " -";
    stat_layer->Print (epochname.str(), false);
    stat_layer->Reset();
	}

	for (NetGraphNode* training_node : graph_.GetTrainingNodes())
		(dynamic_cast<TrainingLayer*>(training_node->layer))->SetTestingMode(false);

  graph_.SetIsTesting(false);

	delete[] loss_sums;
}

void Trainer::Epoch() {
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

  auto t_begin = std::chrono::system_clock::now();

  for (unsigned int i = 0; i < iterations; i++) {
    if ( (50 * i / iterations) > fiftieth) {
      fiftieth = 50 * i / iterations;
      std::cout << "." << std::flush;
    }

    if ( (10 * i / iterations) > tenth) {
      tenth = 10 * i / iterations;
      std::cout << tenth << "0%" << std::flush;
    }

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
  }

  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> t_diff = t_end - t_begin;
  LOGDEBUG << "Training, sps: " <<
          (datum) (sample_count_ * settings_.sbatchsize
                   * first_training_layer_->GetLossSamplingProbability() * iterations)
          / (datum) t_diff.count();

  LOGDEBUG << "Training, tps: " <<
          1000000.0f * (datum) t_diff.count() /
          (datum) (sample_count_ * settings_.sbatchsize
                   * first_training_layer_->GetLossSamplingProbability() * iterations) << " us";
                  
#ifdef BUILD_OPENCL
  LOGDEBUG << "Training, GB/s   up: " << ((datum)CLHelper::bytes_up)/(1073741824.0 * (datum)t_diff.count());
  LOGDEBUG << "Training, GB/s down: " << ((datum)CLHelper::bytes_down)/(1073741824.0 * (datum)t_diff.count());
  CLHelper::bytes_up = 0;
  CLHelper::bytes_down = 0;
#endif

  // Display training epoch_error
	for (unsigned int n = 0; n < graph_.GetLossNodes().size(); n++) {
		LossFunctionLayer* lossfunction_layer = dynamic_cast<LossFunctionLayer*>(graph_.GetLossNodes()[n]->layer);
		LOGINFO << "Training (Epoch " << epoch_ << ", node " << n << ") " << graph_.GetLossNodes()[n]->layer->GetLayerDescription() <<  " lps: " << loss_sums[n] / (datum)(iterations * sample_count_ * settings_.sbatchsize * first_training_layer_->GetLossSamplingProbability());
	}
	LOGINFO << "Training (Epoch " << epoch_ << ") aggregate lps: " << aggregate_loss / (datum)(iterations * sample_count_ * settings_.sbatchsize * first_training_layer_->GetLossSamplingProbability());

	for (unsigned int n = 0; n < graph_.GetStatNodes().size(); n++) {
		StatLayer* stat_layer = dynamic_cast<StatLayer*>(graph_.GetStatNodes()[n]->layer);
    std::stringstream epochname;
    epochname << "Training  - Epoch " << epoch_ << " -";
    stat_layer->Print (epochname.str(), true);
    stat_layer->Reset();
	}

  delete[] loss_sums;
  epoch_++;
}

void Trainer::ApplyGradients (datum lr) {
  unsigned int dp = 0;

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
          layer_lr * (w_gradient / (datum) (sample_count_ * settings_.sbatchsize)) +
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
            if(last_step > 0.001) {
              if(delta > 0.0) {
                step += lr * settings_.eta * delta;
              }
              
              if(delta > (s * last_gradient)) {
                step += settings_.mu * last_step;
              } else {
                step += last_step * delta / (last_gradient - delta);
              }
              
            } else if(last_step < -0.001) {
              if(delta < 0.0) {
                step += lr * settings_.eta * delta;
              }
              
              if(delta < (s * last_gradient)) {
                step += settings_.mu * last_step;
              } else {
                step += last_step * delta / (last_gradient - delta);
              }
            } else {
              step += lr * settings_.eta * delta;
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
