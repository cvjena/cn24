/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file CVLabeledDataLayer.h
 * \class CVLabeledDataLayer
 * \brief This layer outputs labeled data with cross-validation.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_CVLABELEDDATALAYER_H
#define CONV_CVLABELEDDATALAYER_H

#include <vector>
#include <random>


#include "Tensor.h"
#include "CombinedTensor.h"

#include "Layer.h"
#include "TrainingLayer.h"

namespace Conv {

class CVLabeledDataLayer : public Layer, public TrainingLayer {
public:
  /**
   * \brief Creates a CVLabeledDataLayer from data and label Tensors.
   *
   * Note that this constructor _moves_ the tensors!
   *
   * \param data The training data
   * \param labels The corresponding labels
   * \param batch_size Number of samples per batch
   * \param seed Random seed for cross validation
   */
  CVLabeledDataLayer (Tensor& data, Tensor& labels,
                             const unsigned int batch_size = 1,
                             const int seed = 0,
                             const unsigned int split = 2
                            );
  /**
   * \brief Sets the way the dataset is split for training and validation.
   * 
   * \param split Set to n for n-fold cross-validation
   */
  void SetCrossValidationSplit (const unsigned int split);
  
  /**
   * \brief Sets the n-th subset of the dataset as the testing set.
   * 
   * \param part Zero-Indexed subset of the dataset to use for testing
   * (smaller than split)
   */
  void SetCrossValidationTestingSubset (const unsigned int testing_subset);
  
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
  Tensor data_;
  Tensor labels_;
  
  CombinedTensor* data_output_ = nullptr;
  CombinedTensor* label_output_ = nullptr;
  
  // Settings
  unsigned int batch_size_;
  unsigned int split_ = 1;
  unsigned int testing_subset_ = 0;
  bool testing_ = false;
  
  // Random seed
  int seed_;
  std::mt19937 generator_;
  
  // Array containing every sample's cross-validation subset
  unsigned int* subset_ = nullptr;
  
  // Array containing a random permutation of the samples
  std::vector<unsigned int> perm_;
  unsigned int current_element_ = 0;
  
  /**
   * \brief Clears the permutation vector and generates a new one.
   */
  void RedoPermutation();
};

}

#endif
