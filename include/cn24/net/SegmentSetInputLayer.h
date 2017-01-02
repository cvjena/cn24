/*
 * This file is part of the CN24 semantic segmentation software,
 * copyright (C) 2015 Clemens-Alexander Brust (ikosa dot de at gmail dot com).
 *
 * For licensing information, see the LICENSE file included with this project.
 */  
/**
 * @file SegmentSetInputLayer.h
 * @class SegmentSetInputLayer
 * @brief This layer outputs labeled data from a Dataset.
 *
 * @author Clemens-Alexander Brust (ikosa dot de at gmail dot com)
 */

#ifndef CONV_SEGMENTSETINPUTLAYER_H
#define CONV_SEGMENTSETINPUTLAYER_H

#include <vector>
#include <random>
#include <iostream>

#include "Layer.h"
#include "TrainingLayer.h"

#include "../util/SegmentSet.h"
#include "../util/Dataset.h"

namespace Conv {

class SegmentSetInputLayer : public Layer, public TrainingLayer {
public:
  /**
	* @brief Constructs a SegmentSetInputLayer
	*
	* @param dataset The dataset to read images from
	* @param batch_size The parallel minibatch size 
	* @param loss_sampling_p p value for spatial loss sampling
	* @param seed The random seed to use for sample selection and loss sampling
	*/
  SegmentSetInputLayer(JSON configuration, Task task, ClassManager* class_manager, const unsigned int batch_size = 1,
      const unsigned int seed = 0
		    );

  void SetActiveTestingSet(unsigned int index);
  unsigned int GetActiveTestingSet() const { return testing_set_; }

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
  bool ForceLoadDetection(JSON& sample, unsigned int index);
  void ForceWeightsZero();
  void SetTestingMode (bool testing);
  unsigned int GetSamplesInTrainingSet();
  unsigned int GetSamplesInTestingSet();
  unsigned int GetBatchSize();
  unsigned int GetLabelWidth();
  unsigned int GetLabelHeight();

  inline datum GetLossSamplingProbability() { return 1; }

	std::string GetLayerDescription() { return "SegmentSet Input Layer"; }
	void CreateBufferDescriptors(std::vector<NetGraphBuffer>& buffers);
  bool IsGPUMemoryAware();

  Task GetTask() const { return task_; }

  std::vector<SegmentSet*> training_sets_;
  std::vector<datum> training_weights_;
  std::vector<SegmentSet*> staging_sets_;
  std::vector<SegmentSet*> testing_sets_;
  void UpdateDatasets();
private:
  void LoadSampleAugmented(unsigned int sample, const datum x_scale, const datum x_transpose_img, const datum y_scale,
                           const datum y_transpose_img, const bool flip_horizontal, const datum flip_offset);
  void AugmentInPlaceSatExp(unsigned int sample, const datum saturation_factor, const datum exposure_factor);

  datum training_weight_sum_ = 0;
  Task task_;

  // Outputs
  CombinedTensor* data_output_ = nullptr;
  CombinedTensor* label_output_ = nullptr;
  CombinedTensor* helper_output_ = nullptr;
  CombinedTensor* localized_error_output_ = nullptr;
  std::vector<DetectionMetadata> metadata_;

  unsigned int batch_size_;
  unsigned int input_maps_;
  unsigned int input_width_;
  unsigned int input_height_;

  unsigned int elements_training_ = 0;
  unsigned int elements_testing_ = 0;

  bool testing_ = false;
  unsigned int testing_set_ = 0;

  ClassManager* class_manager_;

  // Random generation
  std::mt19937 generator_;
  std::uniform_real_distribution<datum> dist_;

  unsigned int current_element_testing_ = 0;
  
  // Augmentation settings
  datum jitter_;
  datum exposure_;
  datum saturation_;
  int flip_;
  bool do_augmentation_;

  // Augmentation buffers
  std::vector<DetectionMetadata> preaug_metadata_;
  Tensor preaug_data_buffer_;

};

}

#endif
