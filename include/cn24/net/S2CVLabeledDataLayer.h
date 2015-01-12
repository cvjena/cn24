/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file S2CVLabeledDataLayer.h
 * \class S2CVLabeledDataLayer
 * \brief This layer outputs labeled data with cross-validation for semantic
 * segmentation.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_S2CVLABELEDDATALAYER_H
#define CONV_S2CVLABELEDDATALAYER_H

#include <vector>
#include <random>


#include "Tensor.h"
#include "CombinedTensor.h"

#include "Layer.h"
#include "TrainingLayer.h"

namespace Conv {
/*
 * \brief This type is used for function pointers to a localized error function.
 *
 * A localized error function weighs the network's error w.r.t x and y
 * coordinates.
 */
typedef datum (*localized_error_function) (unsigned int, unsigned int);

inline datum DefaultLocalizedErrorFunction (unsigned int x, unsigned int y) {
  return 1.0;
}

class S2CVLabeledDataLayer : public Layer, public TrainingLayer {
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
   * \param split Cross validation split
   * \param resume_from Resume from element
   * \param helper_position Add context information (position)
   * \param randomize_subsets Randomize the subset for each sample
   * \param normalize_mean Substract the mean from each sample
   * \param normalize_stddev Divide each sample by its standard deviation
   */
  S2CVLabeledDataLayer (Tensor& data, Tensor& labels,
                        const unsigned int patchsize_x,
                        const unsigned int patchsize_y,
                        const unsigned int batch_size = 1,
                        const int seed = 0,
                        const unsigned int split = 2,
                        const unsigned int resume_from = 0,
                        const bool helper_position = true,
                        const bool randomize_subsets_ = false,
                        const bool normalize_mean = true,
                        const bool normalize_stddev_ = false,
                        const localized_error_function error_function =
                          DefaultLocalizedErrorFunction
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
  void SetCrossValidationTestingSubset (const int testing_subset);

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

  inline unsigned int current_element() {
    return current_element_;
  }

private:
  // Stored data
  Tensor data_;
  Tensor labels_;

  CombinedTensor* data_output_ = nullptr;
  CombinedTensor* label_output_ = nullptr;
  CombinedTensor* helper_output_ = nullptr;
  CombinedTensor* localized_error_output_ = nullptr;

  // Settings
  unsigned int patchsize_x_;
  unsigned int patchsize_y_;
  unsigned int patches_;
  unsigned int ppi_;
  unsigned int ppr_;
  unsigned int batch_size_;
  unsigned int split_ = 1;
  unsigned int testing_subset_ = 0;
  bool testing_ = false;
  bool select_all_ = false;

  bool randomize_subsets_ = false;
  bool helper_x_ = true;
  bool helper_y_ = true;
  bool normalize_mean_ = true;
  bool normalize_stddev_ = true;

  // Random seed
  int seed_;
  std::mt19937 generator_;

  // Array containing every sample's cross-validation subset
  unsigned int* subset_ = nullptr;

  // Array containing a random permutation of the samples
  std::vector<unsigned int> perm_;
  unsigned int current_element_ = 0;

  unsigned int current_element_testing_ = 0;
  
  // Pointer to the localized error function
  localized_error_function error_function_;

  /**
   * \brief Clears the permutation vector and generates a new one.
   */
  void RedoPermutation();
};

}

#endif
