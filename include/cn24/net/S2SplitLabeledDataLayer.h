/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file S2SplitLabeledDataLayer.h
 * \class S2SplitLabeledDataLayer
 * \brief This layer outputs labeled data with cross-validation for semantic
 * segmentation.
 *
 * \author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_S2SPLITLABELEDDATALAYER_H
#define CONV_S2SPLITLABELEDDATALAYER_H

#include <vector>
#include <random>
#include <iostream>


#include "Tensor.h"
#include "CombinedTensor.h"

#include "Layer.h"
#include "TrainingLayer.h"
#include "S2CVLabeledDataLayer.h"

namespace Conv {
/*
 * \brief This type is used for function pointers to a localized error function.
 *
 * A localized error function weighs the network's error w.r.t x and y
 * coordinates.
 */


class S2SplitLabeledDataLayer : public Layer, public TrainingLayer {
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
  S2SplitLabeledDataLayer (std::istream& training, std::istream& testing,
                        const unsigned int patchsize_x,
                        const unsigned int patchsize_y,
                        const unsigned int batch_size = 1,
                        const int seed = 0,
                        const unsigned int resume_from = 0,
                        const bool helper_position = true,
                        const bool normalize_mean = true,
                        const bool normalize_stddev_ = false,
                        const int ignore_class = -1,
                        const localized_error_function error_function =
                        DefaultLocalizedErrorFunction,
                        const datum* per_class_weights = nullptr
                       );

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

#ifdef BUILD_OPENCL
  bool IsOpenCLAware() { return true; }
#endif

private:
  // Stored data
  Tensor* data_ = nullptr;
  Tensor* labels_ = nullptr;

  CombinedTensor* data_output_ = nullptr;
  CombinedTensor* label_output_ = nullptr;
  CombinedTensor* helper_output_ = nullptr;
  CombinedTensor* localized_error_output_ = nullptr;

  // Settings
  unsigned int patchsize_x_;
  unsigned int patchsize_y_;
  unsigned int batch_size_;
  unsigned int input_maps_;
  unsigned int label_maps_;
  
  unsigned int elements_training_ = 0;
  unsigned int elements_testing_ = 0;
  unsigned int elements_total_ = 0;
  
  unsigned int* first_element_ = nullptr;
  unsigned int* last_element_ = nullptr;
  unsigned int tensors_ = 0;
  
  bool testing_ = false;
  bool select_all_ = false;

  bool randomize_subsets_ = false;
  bool helper_x_ = true;
  bool helper_y_ = true;
  bool normalize_mean_ = true;
  bool normalize_stddev_ = true;
 
  int ignore_class_;

  // Random seed
  int seed_;
  std::mt19937 generator_;

  // Array containing a random permutation of the training samples
  std::vector<unsigned int> perm_;
  unsigned int current_element_ = 0;

  unsigned int current_element_testing_ = 0;
  
  // Pointer to the localized error function
  localized_error_function error_function_;

  // Per-class weights
  const datum* per_class_weights_ = nullptr;

  /**
   * \brief Clears the permutation vector and generates a new one.
   */
  void RedoPermutation();
};

}

#endif
