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

#include "Layer.h"
#include "TrainingLayer.h"

#include "../util/Dataset.h"

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
  DatasetInputLayer(JSON configuration, Dataset* initial_dataset, const unsigned int batch_size = 1,
			const datum loss_sampling_p = 1.0,
      const unsigned int seed = 0
		    );

  void SetActiveTestingDataset(Dataset* dataset);
  Dataset* GetActiveTestingDataset() const { return testing_dataset_; }

  // Implementations for Layer
  bool CreateOutputs (const std::vector< CombinedTensor* >& inputs,
                      std::vector< CombinedTensor* >& outputs);
  bool Connect (const std::vector< CombinedTensor* >& inputs,
                const std::vector< CombinedTensor* >& outputs,
                const NetStatus* net );
  void FeedForward();
  void BackPropagate();

  // Implementations for TrainingLayer
  void SelectAndLoadSamples();
  void SetTestingMode (bool testing);
  unsigned int GetSamplesInTrainingSet();
  unsigned int GetSamplesInTestingSet();
  unsigned int GetBatchSize();
  unsigned int GetLabelWidth();
  unsigned int GetLabelHeight();

  inline datum GetLossSamplingProbability() {
    return loss_sampling_p_;
  }

	std::string GetLayerDescription() { return "Dataset Input Layer"; }
	void CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers);
  bool IsGPUMemoryAware();

  void AddDataset(Dataset* dataset, const datum weight);
  void SetWeight(Dataset* dataset, const datum weight);

  const std::vector<Dataset*>& GetDatasets() const { return datasets_; }
  const std::vector<datum>& GetWeights() const { return weights_; }
private:
  void UpdateDatasets();
  void LoadSampleAugmented(unsigned int sample, const datum x_scale, const datum x_transpose_img, const datum y_scale,
                           const datum y_transpose_img, const bool flip_horizontal, const datum flip_offset);
  void AugmentInPlaceSatExp(unsigned int sample, const datum saturation_factor, const datum exposure_factor);

  std::vector<Dataset*> datasets_;
  std::vector<datum> weights_;
  datum weight_sum_ = 0;
  Dataset* testing_dataset_ = nullptr;

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

  bool testing_ = false;

  datum loss_sampling_p_ = 1.0;

  // Random generation
  std::mt19937 generator_;
  std::uniform_real_distribution<datum> dist_;

  unsigned int current_element_testing_ = 0;
  
  // Metadata buffer
  DatasetMetadataPointer* metadata_buffer_ = nullptr;


  // Augmentation settings
  datum jitter_;
  datum exposure_;
  datum saturation_;
  int flip_;
  bool do_augmentation_;

  // Augmentation buffers
  DatasetMetadataPointer* preaug_metadata_buffer_ = nullptr;
  std::vector<std::vector<BoundingBox>> augmented_boxes_;
  Tensor preaug_data_buffer_;

};

}

#endif
