/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * \file DatasetInputLayer.h
 * \class DatasetInputLayer
 * \brief This layer outputs labeled data from a Dataset.
 *
 * \author Clemens-A. Brust (ikosa.de@gmail.com)
 */

#ifndef CONV_DATASETINPUTLAYER_H
#define CONV_DATASETINPUTLAYER_H

#include <vector>
#include <random>
#include <iostream>

#include "Tensor.h"
#include "CombinedTensor.h"

#include "Layer.h"
#include "TrainingLayer.h"

#include "Dataset.h"

namespace Conv {

class DatasetInputLayer : public Layer, public TrainingLayer {
public:
  DatasetInputLayer(Dataset& dataset, const unsigned int batch_size = 1,
		    const unsigned int seed = 0
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
  Dataset& dataset_;
  
  // Outputs
  CombinedTensor* data_output_ = nullptr;
  CombinedTensor* label_output_ = nullptr;
  CombinedTensor* helper_output_ = nullptr;
  CombinedTensor* localized_error_output_ = nullptr;

  unsigned int batch_size_;
  unsigned int input_maps_;
  unsigned int label_maps_;
  
  unsigned int elements_training_ = 0;
  unsigned int elements_testing_ = 0;
  unsigned int elements_total_ = 0;
  
  unsigned int tensors_ = 0;
  
  bool testing_ = false;

  // Random seed
  int seed_;
  std::mt19937 generator_;

  // Array containing a random permutation of the training samples
  std::vector<unsigned int> perm_;
  unsigned int current_element_ = 0;

  unsigned int current_element_testing_ = 0;
  
  /**
   * \brief Clears the permutation vector and generates a new one.
   */
  void RedoPermutation();
};

}

#endif
