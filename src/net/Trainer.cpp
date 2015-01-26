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

#include "Trainer.h"

namespace Conv {

Trainer::Trainer ( Conv::Net& net, TrainerSettings settings ) :
  net_ ( net ), settings_ ( settings ) {
  LOGDEBUG << "Instance created";

  // We need a training layer to select training samples and some kind of
  // loss function to minimize
  if ( net_.training_layer() == nullptr || net_.lossfunction_layer() == nullptr ) {
    FATAL ( "Net doesn't have training layer or loss function layer!" );
  }

  // Save pointers
  training_layer_ = net_.training_layer();
  lossfunction_layer_ = net.lossfunction_layer();

  // Ask the Net for parameters
  net_.GetParameters ( parameters_ );

  LOGDEBUG << "Optimizing " << parameters_.size() << " sets of parameters.";

  unsigned int w = 0;

  for ( unsigned int p = 0; p < parameters_.size(); p++ ) {
    w += parameters_[p]->data.elements();

    // Allocate Tensors for momentum
    Tensor* last_delta = new Tensor();
    last_delta->Resize ( parameters_[p]->data );
    last_delta->Clear();

    last_deltas_.push_back ( last_delta );
  }

  // Outputs the number of weights
  LOGDEBUG << "Weights: " << w;

  // ..and an overview of the training settings
  LOGINFO << "Training settings: " << settings_;
}

void Trainer::Train ( unsigned int epochs ) {
  net_.SetDropoutEnabled ( true );
  net_.SetTestOnlyStatDisabled ( false );

  for ( unsigned int e = 0; e < epochs; e++ )
    Epoch();

  net_.SetDropoutEnabled ( false );
  net_.SetTestOnlyStatDisabled ( false );
}

datum Trainer::Test() {
  datum loss_sum = 0;
  unsigned int stat_count = net_.stat_layers().size();

  datum* stat_sum = new datum[stat_count];

  for ( unsigned int s = 0; s < stat_count; s++ )
    stat_sum[s] = 0;

  unsigned int batchsize = training_layer_->GetBatchSize();
  unsigned int iterations = ( training_layer_->GetSamplesInTestingSet()
                              / batchsize ) + 1;
  iterations = ( unsigned int ) ( ( ( datum ) iterations ) *
                                  settings_.testing_ratio );

  training_layer_->SetTestingMode ( true );

  LOGDEBUG << "Testing, iterations: " << iterations <<
           ", batch size: " << batchsize;

  auto t_begin = std::chrono::system_clock::now();

  for ( unsigned int i = 0; i < iterations; i++ ) {
    net_.FeedForward();
    loss_sum += lossfunction_layer_->CalculateLossFunction();

    for ( unsigned int s = 0; s < stat_count; s++ ) {
      stat_sum[s] += net_.stat_layers() [s]->CalculateStat();
    }
  }

  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> t_diff = t_end - t_begin;
  LOGINFO << "Testing, sps: " <<
          ( datum ) ( training_layer_->GetBatchSize() * iterations )
          / ( datum ) t_diff.count();

  LOGINFO << "Testing, tps: " <<
          1000000.0f * ( datum ) t_diff.count() /
          ( datum ) ( training_layer_->GetBatchSize() * iterations ) << " us";

  LOGDEBUG << "Testing, loss: " << loss_sum / ( datum ) iterations;

  for ( unsigned int s = 0; s < stat_count; s++ ) {
    LOGRESULT << "Testing, " << net_.stat_layers() [s]->stat_name() <<
              ": " << stat_sum[s] / ( datum ) iterations << LOGRESULTEND;
  }

  if ( net_.binary_stat_layer() != nullptr ) {
    std::stringstream epochname;
    epochname << "Testing  - Epoch " << epoch_ << " -";
    net_.binary_stat_layer()->Print ( epochname.str(), false );
    net_.binary_stat_layer()->Reset();
  }

  if ( net_.confusion_matrix_layer() != nullptr ) {
    std::stringstream epochname;
    epochname << "Testing  - Epoch " << epoch_ << " -";
    net_.confusion_matrix_layer()->Print ( epochname.str(), false );
    net_.confusion_matrix_layer()->Reset();
  }

  delete[] stat_sum;
  return loss_sum / ( datum ) iterations;
}

void Trainer::Epoch() {
  datum epoch_error = 0.0;
  unsigned int stat_count = net_.stat_layers().size();
  datum* stat_sum = new datum[stat_count];
  unsigned int batchsize = training_layer_->GetBatchSize();
  unsigned int iterations =
    settings_.iterations == 0 ?
    training_layer_->GetSamplesInTrainingSet() :
    settings_.iterations;
  iterations = ( unsigned int ) ( ( ( datum ) iterations ) *
                                  settings_.epoch_training_ratio );

  unsigned int fiftieth = 0;
  unsigned int tenth = 0;

  training_layer_->SetTestingMode ( false );

  LOGDEBUG << "Epoch: " << epoch_ << ", it: " << iterations <<
           ", bsize: " << batchsize << ", lr0: " <<
           CalculateLR ( epoch_ * iterations ) << std::endl;


  for ( unsigned int s = 0; s < stat_count; s++ )
    stat_sum[s] = 0;


  auto t_begin = std::chrono::system_clock::now();

  for ( unsigned int i = 0; i < iterations; i++ ) {
    if ( ( 50 * i / iterations ) > fiftieth ) {
      fiftieth = 50 * i / iterations;
      std::cout << "." << std::flush;
    }

    if ( ( 10 * i / iterations ) > tenth ) {
      tenth = 10 * i / iterations;
      std::cout << tenth << "0%" << std::flush;
    }

    net_.FeedForward();

    // Save errors
    epoch_error += lossfunction_layer_->CalculateLossFunction();

    for ( unsigned int s = 0; s < stat_count; s++ ) {
      stat_sum[s] += net_.stat_layers() [s]->CalculateStat();
    }

    // Correct errors
    net_.BackPropagate();

    // Calculate annealed learning rate
    const datum lr =
      CalculateLR ( epoch_ * iterations + i );

    // Apply gradients with new learning rate
    ApplyGradients ( lr );
  }

  auto t_end = std::chrono::system_clock::now();
  std::chrono::duration<double> t_diff = t_end - t_begin;
  LOGINFO << "Training, sps: " <<
          ( datum ) ( training_layer_->GetBatchSize()
          * training_layer_->GetLossSamplingProbability() * iterations )
          / ( datum ) t_diff.count();

  LOGINFO << "Training, tps: " <<
          1000000.0f * ( datum ) t_diff.count() /
          ( datum ) ( training_layer_->GetBatchSize()
          * training_layer_->GetLossSamplingProbability() * iterations ) << " us";

  // Display training epoch_error
  LOGDEBUG << "Training, lps: " << epoch_error / ( datum ) (iterations * batchsize
    * training_layer_->GetLossSamplingProbability());

  for ( unsigned int s = 0; s < stat_count; s++ ) {
    LOGTRESULT << "Training, " << net_.stat_layers() [s]->stat_name() <<
               ": " << stat_sum[s] / ( datum ) iterations << LOGRESULTEND;
  }

  if ( net_.binary_stat_layer() != nullptr ) {
    std::stringstream epochname;
    epochname << "Training - Epoch " << epoch_ << " -";
    net_.binary_stat_layer()->Print ( epochname.str(), true );
    net_.binary_stat_layer()->Reset();
  }
  
  if ( net_.confusion_matrix_layer() != nullptr ) {
    std::stringstream epochname;
    epochname << "Testing  - Epoch " << epoch_ << " -";
    net_.confusion_matrix_layer()->Print ( epochname.str(), false );
    net_.confusion_matrix_layer()->Reset();
  }

  delete[] stat_sum;

  epoch_++;
}

void Trainer::ApplyGradients ( datum lr ) {
  unsigned int dp = 0;

  for ( unsigned int l = 0; l < net_.layers_.size(); l++ ) {
    const datum llr = lr * net_.layers_[l]->local_lr_;

    for ( unsigned int p = 0; p < net_.layers_[l]->parameters().size(); p++ ) {
      CombinedTensor* const param = net_.layers_[l]->parameters_[p];
#ifdef BUILD_OPENCL
      param->data.MoveToCPU();
      param->delta.MoveToCPU();
#endif

      for ( unsigned int w = 0; w < param->data.elements(); w++ ) {
        const datum weight = param->data ( w );
        const datum l1_gradient = ( weight > 0 ) - ( weight < 0 );
        const datum l2_gradient = weight;
        const datum w_gradient = param->delta ( w );

        /*
         * http://www.iro.umontreal.ca/~pift6266/H10/notes/gradient.html
         *
         * This site says that one should average the gradient over
         * the minibatch
         */
        const datum last_delta = ( *last_deltas_[dp] ) ( w );
        const datum delta = llr *
                            ( w_gradient / (datum)training_layer_->GetBatchSize()) +
                              lr * (settings_.l2_weight * l2_gradient +
                              settings_.l1_weight * l1_gradient)
                            ;
        const datum step = delta + settings_.momentum * last_delta;
        param->data[w] -= step;

        // Backup delta
        ( *last_deltas_[dp] ) [w] = step;
      }

      dp++;
    }
  }
}



std::ostream& operator<< ( std::ostream& output,
                           const TrainerSettings settings ) {
  output << "LR: " << settings.learning_rate << ", ";
  output << "GM: " << settings.gamma << ", ";
  output << "EX: " << settings.exponent << ", ";
  output << "L1: " << settings.l1_weight << ", ";
  output << "L2: " << settings.l2_weight << ", ";
  output << "MM: " << settings.momentum;
  return output;
}


}
