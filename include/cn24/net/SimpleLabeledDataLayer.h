/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file SimpleLabeledDataLayer.h
 * \class SimpleLabeledDataLayer
 * \brief This layer outputs user supplied data and labels for training.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_SIMPLELABELEDDATALAYER_H
#define CONV_SIMPLELABELEDDATALAYER_H

#include <vector>
#include <random>


#include "Tensor.h"
#include "CombinedTensor.h"

#include "Layer.h"
#include "TrainingLayer.h"

namespace Conv {

class SimpleLabeledDataLayer : public Layer, public TrainingLayer {
public:
  /**
   * \brief Creates a SimpleLabeledDataLayer from training and testing data.
   *
   * \param training_data Training data Tensor
   * \param training_labels Training label Tensor
   * \param testing_data Testing data Tensor
   * \param testing_labels Testing label Tensor
   * \param batch_size Number of samples per batch
   * \param seed Random seed for permutation
   */
  SimpleLabeledDataLayer (Tensor& training_data, Tensor& training_labels,
                          Tensor& testing_data, Tensor& testing_labels,
                          const unsigned int batch_size = 1,
                          const int seed = 0);
  
  
  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs);
  void FeedForward();
  void BackPropagate();

  // Implementations for TrainingLayer
  void SetTestingMode (bool testing);
  unsigned int GetSamplesInTrainingSet();
  unsigned int GetSamplesInTestingSet();
  unsigned int GetBatchSize();
  
private:
  // Stored data
  Tensor training_data_;
  Tensor training_labels_;
  Tensor testing_data_;
  Tensor testing_labels_;
  
  CombinedTensor* data_output_ = nullptr;
  CombinedTensor* label_output_ = nullptr;
  
  // Settings
  unsigned int batch_size_;
  bool testing_ = false;
  
  // Random seed
  int seed_;
  std::mt19937 generator_;
  
  // Array containing a random permutation of the samples
  std::vector<unsigned int> perm_training_;
  std::vector<unsigned int> perm_testing_;
  unsigned int current_element_testing_ = 0;
  unsigned int current_element_training_ = 0;
  
  /**
   * \brief Clears the permutation vector and generates a new one.
   */
  void RedoPermutationTraining();
  
  /**
   * \brief Clears the permutation vector and generates a new one.
   */
  void RedoPermutationTesting();
};

}


#endif
