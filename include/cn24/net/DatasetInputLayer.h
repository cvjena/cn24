/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file DatasetInputLayer.h
 * @class DatasetInputLayer
 * @brief This layer outputs labeled data from a Dataset.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
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
  /**
	* @brief Constructs a DatasetInputLayer
	*
	* @param dataset The dataset to read images from
	* @param batch_size The parallel minibatch size 
	* @param loss_sampling_p p value for spatial loss sampling
	* @param seed The random seed to use for sample selection and loss sampling
	*/
  DatasetInputLayer(Dataset& dataset, const unsigned int batch_size = 1,
			const datum loss_sampling_p = 1.0,
      const unsigned int seed = 0
		    );
  
  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs,
                const NetStatus* net );
  void FeedForward();
  void BackPropagate();

  // Implementations for TrainingLayer
  void SetTestingMode (bool testing);
  unsigned int GetSamplesInTrainingSet();
  unsigned int GetSamplesInTestingSet();
  unsigned int GetBatchSize();
  unsigned int GetLabelWidth();
  unsigned int GetLabelHeight();

  inline datum GetLossSamplingProbability() {
    return loss_sampling_p_;
  }
  inline unsigned int current_element() {
    return current_element_;
  }

	std::string GetLayerDescription() { return "Dataset Input Layer"; }
	void CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers);
  bool IsOpenCLAware();
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

  datum loss_sampling_p_ = 1.0;

  // Random seed
  int seed_;
  std::mt19937 generator_;
  std::uniform_real_distribution<datum> dist_;

  // Array containing a random permutation of the training samples
  std::vector<unsigned int> perm_;
  unsigned int current_element_ = 0;

  unsigned int current_element_testing_ = 0;
  
  /**
   * @brief Clears the permutation vector and generates a new one.
   */
  void RedoPermutation();
};

}

#endif
